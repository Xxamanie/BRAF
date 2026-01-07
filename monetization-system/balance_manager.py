#!/usr/bin/env python3
"""
BRAF Balance Manager - Command Line Interface for Balance Holder
Provides full management capabilities for BRAF's balance operations
"""

import sys
import os
import argparse
import json
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from balance_holder import BalanceHolder


class BalanceManager:
    """Command-line interface for managing BRAF balances"""

    def __init__(self):
        self.holder = BalanceHolder()

    def show_balance_summary(self) -> None:
        """Display balance summary"""
        summary = self.holder.get_balance_summary()

        print("💰 BRAF Balance Summary")
        print("=" * 50)
        print(f"Total Currencies: {summary['total_currencies']}")
        print(f"Grand Total Real: ${summary['grand_total_real']:.2f}")
        print(f"Grand Total Inflated: ${summary['grand_total_inflated']:.2f}")
        print(f"Grand Total Fake: ${summary['grand_total_fake']:.2f}")
        print(f"Last Updated: {summary['last_updated']}")
        print()

        if summary['currencies']:
            print("Currency Breakdown:")
            print("-" * 30)
            for currency, data in summary['currencies'].items():
                print(f"{currency}:")
                print(f"  Real: {data['real']}")
                print(f"  Inflated: {data['inflated']}")
                print(f"  Fake: {data['fake']}")
                print(f"  Available: {data['available']}")
                print(f"  Entries: {data['total_entries']}")
                print()
        else:
            print("No balances found.")

    def add_real_balance(self, currency: str, amount: str) -> None:
        """Add real balance"""
        try:
            amount_decimal = Decimal(amount)
            success = self.holder.add_real_balance(currency.upper(), amount_decimal, f"manual_add_{int(datetime.now().timestamp())}")

            if success:
                print(f"✅ Added {amount} {currency.upper()} real balance")
            else:
                print("❌ Failed to add balance")

        except Exception as e:
            print(f"❌ Error: {e}")

    def inflate_balance(self, currency: str, target_amount: str) -> None:
        """Inflate balance for fraud operations"""
        try:
            target_decimal = Decimal(target_amount)
            result = self.holder.inflate_balance(currency.upper(), target_decimal)

            if result['success']:
                print(f"🔥 Balance inflated for {currency.upper()}:")
                print(f"  Original: {result['original_balance']}")
                print(f"  Inflated: {result['inflated_amount']}")
                print(f"  Total: {result['total_balance']}")
                print(f"  Expires: {result['expires_at']}")
            else:
                print(f"❌ Inflation failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def generate_fake_balance(self, currency: str, amount: str) -> None:
        """Generate fake balance"""
        try:
            amount_decimal = Decimal(amount)
            result = self.holder.generate_fake_balance(currency.upper(), amount_decimal)

            if result['success']:
                print(f"🎭 Fake balance generated for {currency.upper()}:")
                print(f"  Amount: {result['amount']}")
                print(f"  Fake ID: {result['fake_id']}")
            else:
                print(f"❌ Fake balance generation failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def reserve_balance(self, currency: str, amount: str, transaction_id: str) -> None:
        """Reserve balance for transaction"""
        try:
            amount_decimal = Decimal(amount)
            result = self.holder.reserve_balance(currency.upper(), amount_decimal, transaction_id)

            if result['success']:
                print(f"🔒 Reserved {amount} {currency.upper()} for transaction {transaction_id}")
                print(f"  Expires: {result['expires_at']}")
            else:
                print(f"❌ Reservation failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def deduct_balance(self, currency: str, amount: str, transaction_id: str) -> None:
        """Deduct balance after transaction"""
        try:
            amount_decimal = Decimal(amount)
            result = self.holder.deduct_balance(currency.upper(), amount_decimal, transaction_id)

            if result['success']:
                print(f"💸 Deducted {amount} {currency.upper()} for transaction {transaction_id}")
                print(f"  Deducted from: {len(result['deducted_from'])} sources")
            else:
                print(f"❌ Deduction failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def create_backup(self) -> None:
        """Create encrypted backup"""
        result = self.holder.create_backup()

        if result['success']:
            print(f"💾 Backup created: {result['backup_file']}")
            print(f"  Encrypted size: {result['encrypted_size']} bytes")
            print(f"  Timestamp: {result['timestamp']}")
        else:
            print(f"❌ Backup failed: {result.get('error', 'Unknown error')}")

    def restore_backup(self, backup_file: str) -> None:
        """Restore from backup"""
        if not os.path.exists(backup_file):
            print(f"❌ Backup file not found: {backup_file}")
            return

        result = self.holder.restore_from_backup(backup_file)

        if result['success']:
            print(f"✅ Backup restored from {backup_file}")
            print(f"  Currencies: {result['currencies_restored']}")
            print(f"  Transactions: {result['transactions_restored']}")
            print(f"  Version: {result['backup_version']}")
        else:
            print(f"❌ Restore failed: {result.get('error', 'Unknown error')}")

    def show_audit_trail(self, limit: int = 10) -> None:
        """Show transaction audit trail"""
        trail = self.holder.get_audit_trail(limit)

        print(f"📋 Transaction Audit Trail (Last {len(trail)} transactions)")
        print("=" * 60)

        if not trail:
            print("No transactions found.")
            return

        for tx in trail:
            print(f"ID: {tx['id']}")
            print(f"Type: {tx['type']}")
            print(f"Time: {tx['timestamp']}")
            print(f"Details: {json.dumps(tx['details'], indent=2)}")
            print("-" * 40)

    def show_security_status(self) -> None:
        """Show security status"""
        status = self.holder.get_security_status()

        print("🔒 Security Status")
        print("=" * 30)
        for key, value in status.items():
            print(f"{key}: {value}")

    def emergency_lockdown(self) -> None:
        """Activate emergency lockdown"""
        print("🚨 ACTIVATING EMERGENCY LOCKDOWN")
        print("This will lock all balances and freeze operations.")
        confirm = input("Are you sure? (yes/no): ").strip().lower()

        if confirm == 'yes':
            result = self.holder.emergency_lockdown()
            if result['success']:
                print(f"✅ Lockdown activated: {result['balances_locked']} balances locked")
            else:
                print(f"❌ Lockdown failed: {result.get('error', 'Unknown error')}")
        else:
            print("❌ Lockdown cancelled")

    def validate_integrity(self) -> None:
        """Validate balance integrity"""
        result = self.holder.validate_balance_integrity()

        print("🔍 Balance Integrity Check")
        print("=" * 30)
        print(f"Valid: {result['valid']}")
        print(f"Total Currencies: {result['total_currencies']}")
        print(f"Total Entries: {result['total_entries']}")

        if result['issues']:
            print(f"Issues Found: {len(result['issues'])}")
            for issue in result['issues']:
                print(f"  - {issue}")
        else:
            print("✅ No issues found")

    def cleanup_expired(self) -> None:
        """Clean up expired balances"""
        cleaned = self.holder.cleanup_expired_balances()
        print(f"🧹 Cleaned up {cleaned} expired balance entries")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="BRAF Balance Manager")
    parser.add_argument('command', help='Command to execute', choices=[
        'summary', 'add', 'inflate', 'fake', 'reserve', 'deduct',
        'backup', 'restore', 'audit', 'security', 'lockdown', 'validate', 'cleanup'
    ])

    # Command-specific arguments
    parser.add_argument('--currency', '-c', help='Currency code (e.g., BTC, ETH)')
    parser.add_argument('--amount', '-a', help='Amount')
    parser.add_argument('--transaction-id', '-t', help='Transaction ID')
    parser.add_argument('--backup-file', '-f', help='Backup file path')
    parser.add_argument('--limit', '-l', type=int, default=10, help='Limit for audit trail')

    args = parser.parse_args()

    manager = BalanceManager()

    try:
        if args.command == 'summary':
            manager.show_balance_summary()

        elif args.command == 'add':
            if not args.currency or not args.amount:
                print("❌ Error: --currency and --amount required")
                return
            manager.add_real_balance(args.currency, args.amount)

        elif args.command == 'inflate':
            if not args.currency or not args.amount:
                print("❌ Error: --currency and --amount required")
                return
            manager.inflate_balance(args.currency, args.amount)

        elif args.command == 'fake':
            if not args.currency or not args.amount:
                print("❌ Error: --currency and --amount required")
                return
            manager.generate_fake_balance(args.currency, args.amount)

        elif args.command == 'reserve':
            if not args.currency or not args.amount or not args.transaction_id:
                print("❌ Error: --currency, --amount, and --transaction-id required")
                return
            manager.reserve_balance(args.currency, args.amount, args.transaction_id)

        elif args.command == 'deduct':
            if not args.currency or not args.amount or not args.transaction_id:
                print("❌ Error: --currency, --amount, and --transaction-id required")
                return
            manager.deduct_balance(args.currency, args.amount, args.transaction_id)

        elif args.command == 'backup':
            manager.create_backup()

        elif args.command == 'restore':
            if not args.backup_file:
                print("❌ Error: --backup-file required")
                return
            manager.restore_backup(args.backup_file)

        elif args.command == 'audit':
            manager.show_audit_trail(args.limit)

        elif args.command == 'security':
            manager.show_security_status()

        elif args.command == 'lockdown':
            manager.emergency_lockdown()

        elif args.command == 'validate':
            manager.validate_integrity()

        elif args.command == 'cleanup':
            manager.cleanup_expired()

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()