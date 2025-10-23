"""
Test Installation Script for Boba Shop Simulator
Authors: Jordan Duan and Valerie He
INDENG 174 - Professor Zheng

This script tests if all required dependencies are properly installed.
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'simpy',
        'numpy', 
        'matplotlib',
        'pandas',
        'scipy',
        'seaborn'
    ]
    
    print("Testing package imports...")
    print("-" * 30)
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            failed_imports.append(package)
    
    return failed_imports

def test_simulator_imports():
    """Test if simulator modules can be imported"""
    print("\nTesting simulator module imports...")
    print("-" * 40)
    
    simulator_modules = [
        'boba_simulator',
        'experiment_design', 
        'advanced_analysis'
    ]
    
    failed_modules = []
    
    for module in simulator_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} - {e}")
            failed_modules.append(module)
    
    return failed_modules

def test_basic_functionality():
    """Test basic simulator functionality"""
    print("\nTesting basic functionality...")
    print("-" * 30)
    
    try:
        from boba_simulator import create_baseline_config, BobaShopSimulator
        
        # Test configuration creation
        config = create_baseline_config()
        print("✓ Configuration creation")
        
        # Test simulator initialization
        simulator = BobaShopSimulator(config)
        print("✓ Simulator initialization")
        
        # Test short simulation run
        results = simulator.run_simulation(simulation_time=10)  # Very short run
        print("✓ Basic simulation run")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Boba Shop Simulator - Installation Test")
    print("=" * 50)
    
    # Test package imports
    failed_packages = test_imports()
    
    # Test simulator imports
    failed_modules = test_simulator_imports()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if failed_packages:
        print(f"❌ Missing packages: {', '.join(failed_packages)}")
        print("   Run: pip install -r requirements.txt")
    else:
        print("✅ All required packages installed")
    
    if failed_modules:
        print(f"❌ Module import issues: {', '.join(failed_modules)}")
    else:
        print("✅ All simulator modules imported successfully")
    
    if functionality_ok:
        print("✅ Basic functionality working")
    else:
        print("❌ Basic functionality issues detected")
    
    if not failed_packages and not failed_modules and functionality_ok:
        print("\n🎉 Installation test PASSED!")
        print("You can now run the demo: python demo.py")
    else:
        print("\n⚠️  Installation test FAILED!")
        print("Please fix the issues above before running the simulator.")

if __name__ == "__main__":
    main()
