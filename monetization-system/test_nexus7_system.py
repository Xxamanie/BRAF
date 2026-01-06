#!/usr/bin/env python3
"""
NEXUS7 System Test
Test the complete NEXUS7 research system functionality
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the research directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'research'))

def test_imports():
    """Test all NEXUS7 imports"""
    print("🔬 Testing NEXUS7 System Imports...")
    
    try:
        from research.nexus7_config import nexus7_config, NEXUS7Config
        print("✅ NEXUS7 Config imported successfully")
    except ImportError as e:
        print(f"❌ NEXUS7 Config import failed: {e}")
        return False
    
    try:
        from research.nexus7_integration import nexus7_integration, NEXUS7Integration
        print("✅ NEXUS7 Integration imported successfully")
    except ImportError as e:
        print(f"❌ NEXUS7 Integration import failed: {e}")
        return False
    
    try:
        from research.account_factory import account_factory_research, AccountFactoryResearch
        print("✅ Account Factory Research imported successfully")
    except ImportError as e:
        print(f"❌ Account Factory Research import failed: {e}")
        return False
    
    try:
        from research.survey_research_engine import survey_research_engine, SurveyResearchEngine
        print("✅ Survey Research Engine imported successfully")
    except ImportError as e:
        print(f"❌ Survey Research Engine import failed: {e}")
        return False
    
    try:
        from research.crypto_research_mixer import research_crypto_mixer, CryptoResearchMixer
        print("✅ Crypto Research Mixer imported successfully")
    except ImportError as e:
        print(f"❌ Crypto Research Mixer import failed: {e}")
        return False
    
    try:
        from research.opsec_research_manager import research_opsec_manager, OpSecResearchManager
        print("✅ OpSec Research Manager imported successfully")
    except ImportError as e:
        print(f"❌ OpSec Research Manager import failed: {e}")
        return False
    
    return True

async def test_nexus7_activation():
    """Test NEXUS7 system activation"""
    print("\n🚀 Testing NEXUS7 System Activation...")
    
    try:
        from research.nexus7_integration import nexus7_integration
        
        # Test activation
        activation_result = await nexus7_integration.activate_nexus7()
        
        print(f"✅ NEXUS7 Activation completed")
        print(f"📊 Phases completed: {len(activation_result['phases_completed'])}")
        print(f"🔧 Systems online: {len(activation_result['systems_online'])}")
        print(f"💰 Revenue streams: {len(activation_result['revenue_streams_active'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ NEXUS7 Activation failed: {e}")
        return False

def test_nexus7_config():
    """Test NEXUS7 configuration"""
    print("\n⚙️ Testing NEXUS7 Configuration...")
    
    try:
        from research.nexus7_config import nexus7_config
        
        # Test configuration access
        stats = nexus7_config.get_nexus7_stats()
        print(f"✅ NEXUS7 Stats: {stats['nexus7_mode']}")
        
        # Test fingerprint generation
        fingerprint = nexus7_config.generate_nexus7_fingerprint()
        print(f"✅ Generated fingerprint with ID: {fingerprint['nexus7_id']}")
        
        # Test revenue projections
        revenue = nexus7_config.get_revenue_projections()
        small_op = revenue['small_operation']['weekly_earnings']['total']
        print(f"✅ Revenue projection (small): ${small_op['min']:,} - ${small_op['max']:,}/week")
        
        return True
        
    except Exception as e:
        print(f"❌ NEXUS7 Configuration test failed: {e}")
        return False

async def test_account_factory():
    """Test Account Factory Research"""
    print("\n🏭 Testing Account Factory Research...")
    
    try:
        from research.account_factory import account_factory_research
        
        # Test account creation research
        research_result = await account_factory_research.research_account_creation_patterns(
            platform="swagbucks",
            count=3
        )
        
        print(f"✅ Account research completed")
        print(f"📊 Test accounts: {len(research_result['test_accounts'])}")
        print(f"🔍 Patterns discovered: {len(research_result['patterns_discovered'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Account Factory test failed: {e}")
        return False

async def test_survey_research():
    """Test Survey Research Engine"""
    print("\n📋 Testing Survey Research Engine...")
    
    try:
        from research.survey_research_engine import survey_research_engine
        
        # Test survey optimization research
        test_account = {
            "platform": "swagbucks",
            "identity_data": {
                "age": 35,
                "income_range": "75000-100000",
                "education_level": "bachelor"
            }
        }
        
        research_result = await survey_research_engine.research_survey_optimization(
            platform="swagbucks",
            account=test_account,
            max_surveys=2
        )
        
        print(f"✅ Survey research completed")
        print(f"📊 Surveys researched: {research_result['surveys_researched']}")
        print(f"💰 Research earnings: ${research_result['total_research_earnings']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Survey Research test failed: {e}")
        return False

async def test_crypto_mixer():
    """Test Crypto Research Mixer"""
    print("\n🔄 Testing Crypto Research Mixer...")
    
    try:
        from research.crypto_research_mixer import research_crypto_mixer
        
        # Test cryptocurrency mixing research
        mixing_result = await research_crypto_mixer.research_mix_funds(
            source_wallet="research_wallet_123",
            amount=1.0,
            cryptocurrency="BTC"
        )
        
        print(f"✅ Crypto mixing research completed")
        print(f"🔒 Mixing ID: {mixing_result['mixing_id']}")
        print(f"🎯 Privacy score: {mixing_result['privacy_score']}")
        print(f"💰 Net amount: {mixing_result['net_amount']} BTC")
        
        return True
        
    except Exception as e:
        print(f"❌ Crypto Mixer test failed: {e}")
        return False

async def test_opsec_manager():
    """Test OpSec Research Manager"""
    print("\n🛡️ Testing OpSec Research Manager...")
    
    try:
        from research.opsec_research_manager import research_opsec_manager
        
        # Test security measures
        security_result = research_opsec_manager.enable_research_security()
        
        print(f"✅ OpSec research completed")
        print(f"🔧 Security measures: {len(security_result)}")
        
        # Test cleanup cycle
        cleanup_result = await research_opsec_manager.research_execute_cleanup_cycle("normal")
        
        print(f"✅ Cleanup cycle completed")
        print(f"🧹 Actions taken: {len(cleanup_result['actions_taken'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpSec Manager test failed: {e}")
        return False

async def test_optimization_engine():
    """Test Autonomous Optimization Engine"""
    print("\n🤖 Testing Autonomous Optimization Engine...")
    
    try:
        from research.autonomous_optimization_engine import adaptive_task_engine
        
        # Test engine status
        status = adaptive_task_engine.get_engine_status()
        print(f"✅ Engine status retrieved")
        print(f"📊 Methodologies: {status['methodology_count']}")
        print(f"🎯 Completion rate: {status['average_completion_rate']:.2%}")
        
        # Test optimization cycle
        results = await adaptive_task_engine.execute_optimization_cycle()
        print(f"✅ Optimization cycle completed")
        print(f"🔬 Operations: {results['operations_performed']}")
        print(f"🧬 Variations: {results['successful_variations']}")
        
        # Test task execution
        task_result = await adaptive_task_engine.execute_operational_task(
            target_platform="test_platform",
            task_type="routine_operations"
        )
        print(f"✅ Task execution completed")
        print(f"📈 Task completed: {task_result['completed']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization Engine test failed: {e}")
        return False

async def test_response_optimizer():
    """Test Probabilistic Response Optimizer"""
    print("\n📊 Testing Probabilistic Response Optimizer...")
    
    try:
        from research.probabilistic_response_optimizer import probabilistic_response_optimizer
        
        # Test profile generation
        profile = probabilistic_response_optimizer.generate_consistent_profile()
        print(f"✅ Profile generated")
        print(f"📋 Profile items: {len(profile)}")
        
        # Test demographic response
        age_response = probabilistic_response_optimizer.generate_demographic_response("age_question")
        print(f"✅ Demographic response: {age_response}")
        
        # Test preference response
        tech_response = probabilistic_response_optimizer.generate_preference_response("tech_proficiency")
        print(f"✅ Preference response: {tech_response}")
        
        # Test learning from feedback
        probabilistic_response_optimizer.learn_from_outcome("age_question", age_response, True)
        print(f"✅ Learning from feedback completed")
        
        # Test optimization statistics
        stats = probabilistic_response_optimizer.get_optimization_statistics()
        print(f"✅ Statistics retrieved")
        print(f"📈 Total responses: {stats.get('total_responses', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Response Optimizer test failed: {e}")
        return False

def test_system_status():
    """Test complete system status"""
    print("\n📊 Testing Complete System Status...")
    
    try:
        from research.nexus7_integration import nexus7_integration
        
        # Get system status
        status = nexus7_integration.get_nexus7_status()
        
        print(f"✅ System Status Retrieved")
        print(f"🔧 NEXUS7 Active: {status['nexus7_active']}")
        print(f"🕵️ Stealth Mode: {status['stealth_mode']}")
        print(f"⚡ Operations Running: {status['operations_running']}")
        
        return True
        
    except Exception as e:
        print(f"❌ System Status test failed: {e}")
        return False

async def main():
    """Run all NEXUS7 system tests"""
    print("=" * 60)
    print("🔬 NEXUS7 RESEARCH SYSTEM TEST SUITE")
    print("=" * 60)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_nexus7_config),
        ("System Status Tests", test_system_status),
        ("NEXUS7 Activation", test_nexus7_activation),
        ("Account Factory Research", test_account_factory),
        ("Survey Research Engine", test_survey_research),
        ("Crypto Research Mixer", test_crypto_mixer),
        ("OpSec Research Manager", test_opsec_manager),
        ("Autonomous Optimization Engine", test_optimization_engine),
        ("Probabilistic Response Optimizer", test_response_optimizer),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print("🔬 NEXUS7 TEST RESULTS")
    print("=" * 60)
    print(f"✅ Tests Passed: {passed}")
    print(f"❌ Tests Failed: {failed}")
    print(f"📊 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    print(f"⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! NEXUS7 SYSTEM IS READY!")
        print("🚀 You can now proceed with live operations.")
    else:
        print(f"\n⚠️ {failed} tests failed. Please review the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    asyncio.run(main())
