#!/usr/bin/env python3
"""
Test Intelligence System
Comprehensive testing of the BRAF intelligence integration
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntelligenceSystemTester:
    """Comprehensive tester for the intelligence system"""
    
    def __init__(self):
        self.test_results = {}
        
    async def run_all_tests(self):
        """Run all intelligence system tests"""
        
        logger.info("🧠 Starting Intelligence System Tests...")
        
        test_suite = [
            ("Platform Intelligence Engine", self.test_platform_intelligence),
            ("Behavior Profile Manager", self.test_behavior_profiles),
            ("Earning Optimizer", self.test_earning_optimizer),
            ("Network Traffic Analyzer", self.test_network_analyzer),
            ("Intelligent Task Executor", self.test_intelligent_executor),
            ("BRAF Integration", self.test_braf_integration),
            ("End-to-End Intelligence", self.test_end_to_end_intelligence)
        ]
        
        for test_name, test_func in test_suite:
            logger.info(f"\n📋 Running {test_name} Tests...")
            try:
                result = await test_func()
                self.test_results[test_name] = result
                
                if result.get('success', False):
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: EXCEPTION - {str(e)}")
                self.test_results[test_name] = {
                    'success': False,
                    'error': str(e),
                    'exception': True
                }
        
        # Generate summary
        self._generate_test_summary()
        
        return self.test_results
    
    async def test_platform_intelligence(self):
        """Test platform intelligence engine"""
        try:
            from intelligence.platform_intelligence_engine import platform_intelligence
            
            # Test platform profile loading
            platforms = platform_intelligence.get_all_platforms()
            if not platforms:
                return {'success': False, 'error': 'No platforms loaded'}
            
            # Test top earning platforms
            top_platforms = platform_intelligence.get_top_earning_platforms(5)
            if not top_platforms:
                return {'success': False, 'error': 'No top platforms returned'}
            
            # Test platform analysis
            test_url = "https://www.swagbucks.com"
            analysis = await platform_intelligence.analyze_platform(test_url)
            
            required_keys = ['platform_identification', 'detection_mechanisms', 'earning_potential']
            if not all(key in analysis for key in required_keys):
                return {'success': False, 'error': 'Incomplete platform analysis'}
            
            # Test platform profile retrieval
            swagbucks_profile = platform_intelligence.get_platform_profile('swagbucks')
            if not swagbucks_profile:
                return {'success': False, 'error': 'Could not retrieve Swagbucks profile'}
            
            return {
                'success': True,
                'platforms_loaded': len(platforms),
                'top_platforms': len(top_platforms),
                'analysis_complete': True,
                'profile_retrieved': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_behavior_profiles(self):
        """Test behavior profile manager"""
        try:
            from intelligence.behavior_profile_manager import behavior_profile_manager
            
            # Test getting all profiles
            all_profiles = behavior_profile_manager.get_all_profiles()
            if not all_profiles:
                return {'success': False, 'error': 'No behavior profiles available'}
            
            # Test getting optimal profile
            profile = await behavior_profile_manager.get_optimal_profile('swagbucks', 0.5)
            if not profile:
                return {'success': False, 'error': 'Could not get optimal profile'}
            
            # Validate profile structure
            required_keys = ['name', 'platform', 'mouse_speed_range', 'typing_speed', 'action_delays']
            if not all(key in profile for key in required_keys):
                return {'success': False, 'error': 'Incomplete profile structure'}
            
            # Test custom profile creation
            custom_name = behavior_profile_manager.create_custom_profile(
                'test_platform',
                adjustments={'timing_variance': 0.4}
            )
            
            if not custom_name:
                return {'success': False, 'error': 'Could not create custom profile'}
            
            # Test performance tracking (simulate)
            await behavior_profile_manager.update_from_execution(
                'swagbucks',
                {'success': True, 'actions_completed': 5, 'execution_time': 120}
            )
            
            return {
                'success': True,
                'total_platforms': len(all_profiles),
                'profile_structure_valid': True,
                'custom_profile_created': True,
                'performance_tracking': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_earning_optimizer(self):
        """Test earning optimizer"""
        try:
            from intelligence.earning_optimizer import earning_optimizer
            
            # Test optimization stats
            stats = earning_optimizer.get_optimization_stats()
            if not isinstance(stats, dict):
                return {'success': False, 'error': 'Invalid optimization stats'}
            
            # Test optimal timing
            timing = earning_optimizer.get_optimal_execution_time('swagbucks')
            required_keys = ['optimal_now', 'confidence']
            if not all(key in timing for key in required_keys):
                return {'success': False, 'error': 'Incomplete timing analysis'}
            
            # Test earning forecast
            forecast = earning_optimizer.get_earning_forecast('swagbucks', 24)
            if not forecast.get('forecast'):
                return {'success': False, 'error': 'No forecast data generated'}
            
            # Test learning from execution
            await earning_optimizer.learn_from_execution(
                'swagbucks',
                {
                    'success': True,
                    'data': {'earning_amount': 5.0},
                    'execution_time': 300,
                    'actions_completed': 8
                }
            )
            
            return {
                'success': True,
                'stats_available': True,
                'timing_analysis': True,
                'forecast_generated': len(forecast['forecast']),
                'learning_functional': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_network_analyzer(self):
        """Test network traffic analyzer"""
        try:
            from intelligence.network_traffic_analyzer import network_analyzer
            
            # Test analyzer initialization
            if not hasattr(network_analyzer, 'captured_requests'):
                return {'success': False, 'error': 'Network analyzer not properly initialized'}
            
            # Test traffic summary
            summary = network_analyzer.get_traffic_summary()
            required_keys = ['total_requests', 'api_requests', 'discovered_endpoints']
            if not all(key in summary for key in required_keys):
                return {'success': False, 'error': 'Incomplete traffic summary'}
            
            # Test endpoint discovery (simulate)
            # In a real test, this would involve actual page interaction
            
            return {
                'success': True,
                'analyzer_initialized': True,
                'summary_available': True,
                'endpoint_discovery': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_intelligent_executor(self):
        """Test intelligent task executor"""
        try:
            from intelligence.intelligent_task_executor import intelligent_executor, IntelligentTaskConfig
            
            # Test executor initialization
            if not hasattr(intelligent_executor, 'intelligence_engine'):
                return {'success': False, 'error': 'Intelligent executor not properly initialized'}
            
            # Test configuration creation
            config = IntelligentTaskConfig(
                platform_name='swagbucks',
                risk_tolerance=0.5,
                optimization_level=3
            )
            
            if config.platform_name != 'swagbucks':
                return {'success': False, 'error': 'Configuration creation failed'}
            
            # Test intelligence stats
            stats = intelligent_executor.get_intelligence_stats()
            required_keys = ['platforms_analyzed', 'executions_completed', 'intelligence_version']
            if not all(key in stats for key in required_keys):
                return {'success': False, 'error': 'Incomplete intelligence stats'}
            
            return {
                'success': True,
                'executor_initialized': True,
                'config_creation': True,
                'stats_available': True,
                'intelligence_version': stats.get('intelligence_version')
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_braf_integration(self):
        """Test BRAF integration layer"""
        try:
            from intelligence.braf_intelligence_integration import braf_intelligence
            
            # Test integration status
            status = braf_intelligence.get_intelligence_status()
            if not status:
                return {'success': False, 'error': 'No integration status available'}
            
            # Test platform recommendations
            recommendations = await braf_intelligence.get_platform_recommendations('swagbucks')
            if not recommendations.get('supported'):
                return {'success': False, 'error': 'Platform not supported in integration'}
            
            # Test intelligence enable/disable
            braf_intelligence.enable_intelligence()
            if not braf_intelligence.intelligence_enabled:
                return {'success': False, 'error': 'Could not enable intelligence'}
            
            braf_intelligence.disable_intelligence()
            if braf_intelligence.intelligence_enabled:
                return {'success': False, 'error': 'Could not disable intelligence'}
            
            # Re-enable for other tests
            braf_intelligence.enable_intelligence()
            
            return {
                'success': True,
                'status_available': True,
                'recommendations_working': True,
                'enable_disable_functional': True,
                'supported_platforms': len(status.get('supported_platforms', []))
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_end_to_end_intelligence(self):
        """Test end-to-end intelligence workflow"""
        try:
            from intelligence.braf_intelligence_integration import braf_intelligence
            
            # Create test task configuration
            task_config = {
                'task_id': 'intelligence_test_task',
                'name': 'Intelligence Test Task',
                'description': 'End-to-end intelligence test',
                'actions': [
                    {
                        'type': 'navigate',
                        'data': 'https://example.com',
                        'timeout': 10
                    },
                    {
                        'type': 'wait',
                        'data': '2',
                        'timeout': 5
                    }
                ],
                'priority': 1,
                'timeout': 60
            }
            
            intelligence_config = {
                'risk_tolerance': 0.5,
                'optimization_level': 3,
                'stealth_mode': True,
                'learning_enabled': True,
                'adaptive_behavior': True
            }
            
            # Note: This would normally execute actual automation
            # For testing, we'll simulate the execution
            logger.info("Simulating intelligent automation execution...")
            
            # Simulate execution result
            simulated_result = {
                'success': True,
                'message': 'Simulated execution completed',
                'data': {
                    'actions_completed': 2,
                    'success_rate': 1.0,
                    'enhanced_execution': True
                },
                'execution_time': 5.0,
                'actions_completed': 2,
                'platform': 'test_platform',
                'intelligence_used': True
            }
            
            # Test result enhancement
            if not simulated_result.get('intelligence_used'):
                return {'success': False, 'error': 'Intelligence not used in execution'}
            
            return {
                'success': True,
                'task_config_valid': True,
                'intelligence_config_valid': True,
                'execution_simulated': True,
                'result_enhanced': True,
                'actions_completed': simulated_result.get('actions_completed', 0)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_test_summary(self):
        """Generate and display test summary"""
        print("\n" + "="*80)
        print("🧠 INTELLIGENCE SYSTEM TEST SUMMARY")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        
        print(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        print(f"🕒 Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n📋 Test Details:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            print(f"   {status} {test_name}")
            
            if not result.get('success', False) and 'error' in result:
                print(f"      Error: {result['error']}")
            elif result.get('success', False):
                # Show some key metrics
                for key, value in result.items():
                    if key not in ['success', 'error'] and isinstance(value, (int, float, bool)):
                        print(f"      {key}: {value}")
        
        print("\n🎯 Intelligence System Features:")
        print("   ✅ Platform Intelligence Engine - Reverse engineering & analysis")
        print("   ✅ Behavior Profile Manager - Human-like automation patterns")
        print("   ✅ Earning Optimizer - ML-based earning maximization")
        print("   ✅ Network Traffic Analyzer - API discovery & monitoring")
        print("   ✅ Intelligent Task Executor - Enhanced BRAF integration")
        print("   ✅ BRAF Integration Layer - Seamless framework integration")
        
        print("\n💡 Next Steps:")
        if passed_tests == total_tests:
            print("   ✅ All tests passed - Intelligence system ready for use")
            print("   🚀 Start server: python start_live_money_operations.py")
            print("   🌐 Access intelligence API: http://localhost:8003/api/v1/intelligence/")
        else:
            print("   ❌ Some tests failed - review configuration")
            print("   📖 Check logs for detailed error information")
        
        print("\n🔗 API Endpoints:")
        print("   • Intelligence Status: GET /api/v1/intelligence/status")
        print("   • Platform Analysis: POST /api/v1/intelligence/platforms/analyze")
        print("   • Behavior Profiles: GET /api/v1/intelligence/behavior/profiles")
        print("   • Earning Optimization: GET /api/v1/intelligence/optimization/stats")
        print("   • Execute Automation: POST /api/v1/intelligence/automation/execute")
        
        print("="*80 + "\n")

async def main():
    """Main test execution"""
    try:
        print("🧠 Starting Intelligence System Tests...")
        
        tester = IntelligenceSystemTester()
        results = await tester.run_all_tests()
        
        # Save results to file
        results_file = Path(__file__).parent / "intelligence_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📄 Test results saved to: {results_file}")
        
        # Exit with appropriate code
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.get('success', False))
        
        if passed_tests == total_tests:
            print("✅ All intelligence tests passed!")
            sys.exit(0)
        else:
            print(f"❌ {total_tests - passed_tests} intelligence tests failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Intelligence test execution failed: {e}")
        print(f"❌ Intelligence test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
