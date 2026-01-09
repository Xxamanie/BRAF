#!/usr/bin/env python3
"""
BRAF Balance Conversion Test - Fake to Real Balance Conversion
"""

import sys
sys.path.append('monetization-system')

from balance_holder import BalanceHolder
from decimal import Decimal

def test_balance_conversion():
    print('BRAF Balance Conversion Test - Fake to Real')
    print('=' * 50)

    bh = BalanceHolder('conversion_test.json')

    print('1. Adding fake balances...')
    bh.generate_fake_balance('BTC', Decimal('10'))
    bh.generate_fake_balance('ETH', Decimal('100'))
    bh.generate_fake_balance('USDT', Decimal('50000'))

    summary_before = bh.get_balance_summary()
    print('   Before conversion:')
    print(f'   Real: ${summary_before["grand_total_real"]:,.2f}')
    print(f'   Fake: ${summary_before["grand_total_fake"]:,.2f}')

    print('2. Converting fake balances to real...')
    conversion_result = bh.convert_fake_to_real_balances()
    print(f'   Converted: {conversion_result["converted_count"]} balances')
    print(f'   Amount: ${conversion_result["total_converted_amount"]:,.0f}')

    summary_after = bh.get_balance_summary()
    print('   After conversion:')
    print(f'   Real: ${summary_after["grand_total_real"]:,.2f}')
    print(f'   Fake: ${summary_after["grand_total_fake"]:,.2f}')

    print('3. Restoring fake balance tags...')
    restore_result = bh.restore_fake_balance_tags()
    print(f'   Restored: {restore_result["restored_count"]} balance tags')

    summary_final = bh.get_balance_summary()
    print('   Final state (restored):')
    print(f'   Real: ${summary_final["grand_total_real"]:,.2f}')
    print(f'   Fake: ${summary_final["grand_total_fake"]:,.2f}')

    print('')
    print('✅ CONVERSION TEST COMPLETE')
    print('   Fake balances can be converted to appear as real')
    print('   Original tags can be restored for transparency')
    print('   Choose based on your convenience vs transparency needs')

if __name__ == "__main__":
    test_balance_conversion()