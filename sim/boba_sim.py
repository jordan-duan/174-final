import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import simpy


def _lognormal_params(mean: float, cv: float) -> Tuple[float, float]:
    if mean <= 0:
        return 0.0, 0.0
    cv2 = cv * cv
    sigma2 = math.log(cv2 + 1.0)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean) - 0.5 * sigma2
    return mu, sigma


def sample_lognormal(rng: random.Random, mean: float, cv: float) -> float:
    if mean <= 0:
        return 0.0
    mu, sigma = _lognormal_params(mean, cv)
    return rng.lognormvariate(mu, sigma)


@dataclass
class StationService:
    mean: float  # seconds
    cv: float


@dataclass
class DrinkType:
    name: str
    p: float
    needs_pearls: bool


@dataclass
class Policy:
    cashier_cap: int = 1
    barista_cap: int = 1
    sealer_cap: int = 1
    auto_sealer: bool = False
    seating_congestion: bool = False


@dataclass
class PearlInventory:
    initial: int = 200  # cups worth
    capacity: int = 10000
    reorder_point: int = 100
    batch: int = 150
    cook_time_sec: float = 20 * 60  # 20 minutes


@dataclass
class Config:
    # arrivals
    arrival_rate_per_min: float = 1.2  # lambda (customers per minute)
    sim_duration_min: float = 240.0  # 4 hours
    warmup_min: float = 30.0

    # services (seconds)
    cashier: StationService = field(default_factory=lambda: StationService(mean=20.0, cv=0.30))
    barista: StationService = field(default_factory=lambda: StationService(mean=50.0, cv=0.50))
    sealer_manual: StationService = field(default_factory=lambda: StationService(mean=12.0, cv=0.20))
    sealer_auto: StationService = field(default_factory=lambda: StationService(mean=8.0, cv=0.15))

    # drinks
    drinks: List[DrinkType] = field(
        default_factory=lambda: [
            DrinkType("milk_tea_boba", 0.50, True),
            DrinkType("fruit_tea", 0.20, False),
            DrinkType("milk_tea_no_boba", 0.20, False),
            DrinkType("specialty_boba", 0.10, True),
        ]
    )

    # policy & inventory
    policy: Policy = field(default_factory=Policy)
    pearls: PearlInventory = field(default_factory=PearlInventory)


@dataclass
class Metrics:
    cycle_times: List[float] = field(default_factory=list)
    wait_cashier: List[float] = field(default_factory=list)
    wait_barista: List[float] = field(default_factory=list)
    wait_sealer: List[float] = field(default_factory=list)
    service_cashier: List[float] = field(default_factory=list)
    service_barista: List[float] = field(default_factory=list)
    service_sealer: List[float] = field(default_factory=list)
    pearl_stockouts: int = 0
    customers_completed: int = 0
    customers_arrived: int = 0
    busy_cashier_sec: float = 0.0
    busy_barista_sec: float = 0.0
    busy_sealer_sec: float = 0.0


class BobaShopSim:
    def __init__(self, cfg: Config, seed: Optional[int] = None):
        self.cfg = cfg
        self.env = simpy.Environment()
        self.rng = random.Random(seed)

        # resources
        self.cashier = simpy.Resource(self.env, capacity=cfg.policy.cashier_cap)
        self.baristas = simpy.Resource(self.env, capacity=cfg.policy.barista_cap)
        self.sealers = simpy.Resource(self.env, capacity=cfg.policy.sealer_cap)

        # pearls as a container
        self.pearls = simpy.Container(self.env, capacity=cfg.pearls.capacity, init=cfg.pearls.initial)
        self.batch_in_progress = False

        self.metrics = Metrics()
        self.warmup_sec = cfg.warmup_min * 60.0
        self.run_until_sec = (cfg.warmup_min + cfg.sim_duration_min) * 60.0

        # choose drink distribution CDF
        total_p = sum(d.p for d in cfg.drinks)
        cum = 0.0
        self.drink_cdf: List[Tuple[float, DrinkType]] = []
        for d in cfg.drinks:
            cum += d.p / total_p
            self.drink_cdf.append((cum, d))

    def draw_drink(self) -> DrinkType:
        u = self.rng.random()
        for c, d in self.drink_cdf:
            if u <= c:
                return d
        return self.drink_cdf[-1][1]

    def interarrival(self) -> float:
        # Poisson process with rate per minute -> exponential in seconds
        lam_per_sec = self.cfg.arrival_rate_per_min / 60.0
        return self.rng.expovariate(lam_per_sec)

    def service_time(self, station: str) -> float:
        if station == "cashier":
            s = self.cfg.cashier
        elif station == "barista":
            s = self.cfg.barista
        elif station == "sealer":
            s = self.cfg.sealer_auto if self.cfg.policy.auto_sealer else self.cfg.sealer_manual
        else:
            return 0.0
        return sample_lognormal(self.rng, s.mean, s.cv)

    def cooker(self):
        while True:
            # monitor inventory
            if (not self.batch_in_progress) and self.pearls.level <= self.cfg.pearls.reorder_point:
                self.batch_in_progress = True
                # cook time
                yield self.env.timeout(self.cfg.pearls.cook_time_sec)
                yield self.pearls.put(self.cfg.pearls.batch)
                self.batch_in_progress = False
            else:
                # check periodically
                yield self.env.timeout(10.0)

    def process_customer(self, idx: int):
        arrival = self.env.now
        self.metrics.customers_arrived += 1

        # cashier
        with self.cashier.request() as req:
            t0 = self.env.now
            yield req
            wait_c = self.env.now - t0
            svc_c = self.service_time("cashier")
            yield self.env.timeout(svc_c)

        # barista
        drink = self.draw_drink()
        pearls_needed = 1 if drink.needs_pearls else 0
        if pearls_needed > 0:
            # try to get pearls; if not enough, will wait (stockout)
            if self.pearls.level < pearls_needed:
                self.metrics.pearl_stockouts += 1
            yield self.pearls.get(pearls_needed)

        with self.baristas.request() as req_b:
            t1 = self.env.now
            yield req_b
            wait_b = self.env.now - t1
            svc_b = self.service_time("barista")
            yield self.env.timeout(svc_b)

        # sealer
        with self.sealers.request() as req_s:
            t2 = self.env.now
            yield req_s
            wait_s = self.env.now - t2
            svc_s = self.service_time("sealer")
            yield self.env.timeout(svc_s)

        # seating congestion at pickup
        if self.cfg.policy.seating_congestion:
            dwell = sample_lognormal(self.rng, 8.0, 0.5)  # extra hand-off congestion seconds
            yield self.env.timeout(dwell)

        depart = self.env.now

        # record metrics if after warmup
        if depart >= self.warmup_sec:
            self.metrics.cycle_times.append(depart - arrival)
            self.metrics.wait_cashier.append(wait_c)
            self.metrics.wait_barista.append(wait_b)
            self.metrics.wait_sealer.append(wait_s)
            self.metrics.service_cashier.append(svc_c)
            self.metrics.service_barista.append(svc_b)
            self.metrics.service_sealer.append(svc_s)
            self.metrics.customers_completed += 1

        # accumulate busy time regardless of warmup to estimate utilization over horizon
        self.metrics.busy_cashier_sec += svc_c
        self.metrics.busy_barista_sec += svc_b
        self.metrics.busy_sealer_sec += svc_s

    def arrivals(self):
        i = 0
        while True:
            i += 1
            self.env.process(self.process_customer(i))
            ia = self.interarrival()
            yield self.env.timeout(ia)

    def run(self):
        self.env.process(self.arrivals())
        self.env.process(self.cooker())
        self.env.run(until=self.run_until_sec)


def summarize(cfg: Config, m: Metrics) -> Dict[str, float]:
    sim_horizon_sec = (cfg.warmup_min + cfg.sim_duration_min) * 60.0
    # throughput per hour (post-warmup)
    hours_observed = cfg.sim_duration_min / 60.0
    th = (m.customers_completed / hours_observed) if hours_observed > 0 else 0.0

    def mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    util_cashier = m.busy_cashier_sec / (sim_horizon_sec * cfg.policy.cashier_cap)
    util_barista = m.busy_barista_sec / (sim_horizon_sec * cfg.policy.barista_cap)
    util_sealer = m.busy_sealer_sec / (sim_horizon_sec * cfg.policy.sealer_cap)

    return {
        "arrivals": float(m.customers_arrived),
        "completions": float(m.customers_completed),
        "throughput_per_hour": th,
        "mean_cycle_time_sec": mean(m.cycle_times),
        "mean_wait_cashier_sec": mean(m.wait_cashier),
        "mean_wait_barista_sec": mean(m.wait_barista),
        "mean_wait_sealer_sec": mean(m.wait_sealer),
        "util_cashier": util_cashier,
        "util_barista": util_barista,
        "util_sealer": util_sealer,
        "pearl_stockouts": float(m.pearl_stockouts),
    }


def run_once(cfg: Config, seed: Optional[int] = None) -> Tuple[Metrics, Dict[str, float]]:
    sim = BobaShopSim(cfg, seed=seed)
    sim.run()
    summary = summarize(cfg, sim.metrics)
    return sim.metrics, summary


def run_replications(cfg: Config, reps: int = 30, seed: Optional[int] = None) -> Dict[str, float]:
    rng = random.Random(seed)
    summaries: List[Dict[str, float]] = []
    for _ in range(reps):
        s = rng.randrange(10**9)
        _, summ = run_once(cfg, seed=s)
        summaries.append(summ)

    keys = summaries[0].keys() if summaries else []
    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    agg: Dict[str, float] = {}
    for k in keys:
        agg[f"mean_{k}"] = mean([d[k] for d in summaries])
    return agg

