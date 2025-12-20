#!/usr/bin/env python3
"""
Toggle between free beta mode and paid subscription mode
"""

import os
import sys

def update_env_file(enable_paid=False):
    """Update .env file to enable/disable paid tiers"""
    env_path = ".env"
    
    if not os.path.exists(env_path):
        print("❌ .env file not found")
        return False
    
    # Read current content
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update relevant lines
    updated_lines = []
    for line in lines:
        if line.startswith('FREE_BETA_MODE='):
            updated_lines.append(f'FREE_BETA_MODE={"false" if enable_paid else "true"}\n')
        elif line.startswith('ENABLE_PAID_TIERS='):
            updated_lines.append(f'ENABLE_PAID_TIERS={"true" if enable_paid else "false"}\n')
        else:
            updated_lines.append(line)
    
    # Write back
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    return True

def update_config_file(enable_paid=False):
    """Update config.py to enable/disable paid tiers"""
    config_path = "config.py"
    
    if not os.path.exists(config_path):
        print("❌ config.py file not found")
        return False
    
    # Read current content
    with open(config_path, 'r') as f:
        content = f.read()
    
    if enable_paid:
        # Enable paid tiers by uncommenting them
        content = content.replace('        # "basic": {', '        "basic": {')
        content = content.replace('        # "pro": {', '        "pro": {')
        content = content.replace('        # "enterprise": {', '        "enterprise": {')
        content = content.replace('#     "price":', '            "price":')
        content = content.replace('#     "templates":', '            "templates":')
        content = content.replace('#     "max_automations":', '            "max_automations":')
        content = content.replace('#     "daily_earnings_limit":', '            "daily_earnings_limit":')
        content = content.replace('#     "features":', '            "features":')
        content = content.replace('# }', '        }')
    else:
        # Disable paid tiers by commenting them out
        content = content.replace('        "basic": {', '        # "basic": {')
        content = content.replace('        "pro": {', '        # "pro": {')
        content = content.replace('        "enterprise": {', '        # "enterprise": {')
    
    # Write back
    with open(config_path, 'w') as f:
        f.write(content)
    
    return True

def main():
    """Main function"""
    if len(sys.argv) != 2 or sys.argv[1] not in ['free', 'paid']:
        print("Usage: python toggle_monetization.py [free|paid]")
        print("  free  - Enable free beta mode (all features free)")
        print("  paid  - Enable paid subscription tiers")
        return
    
    enable_paid = sys.argv[1] == 'paid'
    mode = "PAID SUBSCRIPTION" if enable_paid else "FREE BETA"
    
    print(f"🔄 Switching to {mode} mode...")
    
    # Update configuration files
    if update_env_file(enable_paid) and update_config_file(enable_paid):
        print(f"✅ Successfully switched to {mode} mode!")
        
        if enable_paid:
            print("\n💰 PAID SUBSCRIPTION MODE ENABLED")
            print("📋 Features:")
            print("  • Basic: $99/month - 5 automations, $50/day limit")
            print("  • Pro: $299/month - 20 automations, $200/day limit") 
            print("  • Enterprise: $999/month - 100 automations, $1000/day limit")
            print("\n⚠️  Remember to:")
            print("  • Configure Stripe API keys")
            print("  • Update payment processing")
            print("  • Test subscription flows")
        else:
            print("\n🎉 FREE BETA MODE ENABLED")
            print("📋 Features:")
            print("  • All users get full access")
            print("  • Unlimited automations")
            print("  • $1000/day earning limit")
            print("  • All templates and features")
            print("\n💡 Perfect for:")
            print("  • Beta testing")
            print("  • User acquisition")
            print("  • Feature validation")
        
        print(f"\n🔄 Restart the server to apply changes:")
        print("   python run_server.py")
        
    else:
        print("❌ Failed to update configuration files")

if __name__ == "__main__":
    main()