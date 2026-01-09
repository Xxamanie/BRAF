"""
AML BYPASS MODULE
Implements Anti-Money Laundering monitoring evasion
"""

import random
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import hashlib


class TransactionStructuringEngine:
    """
    Implements transaction structuring to avoid AML triggers
    """

    def __init__(self):
        self.aml_thresholds = {
            'single_transaction': 10000,  # $10k SAR threshold
            'daily_total': 25000,  # Daily accumulation trigger
            'monthly_total': 100000,  # Monthly pattern trigger
            'peer_transfers': 5000,  # Frequent same-peer triggers
            'round_amounts': True,  # Round amounts trigger AML
        }

    def structure_transaction(self, amount: float, purpose: str = 'normal') -> List[Dict[str, Any]]:
        """
        Structure a large transaction into multiple smaller ones to avoid AML triggers
        """
        if amount <= self.aml_thresholds['single_transaction'] * 0.8:
            # No structuring needed
            return [{
                'amount': amount,
                'delay_minutes': 0,
                'purpose': purpose,
                'structured': False
            }]

        # Calculate optimal structure
        max_per_transaction = self.aml_thresholds['single_transaction'] * random.uniform(0.3, 0.7)
        num_transactions = max(2, int(amount / max_per_transaction) + 1)

        # Distribute amount across transactions with slight variations
        base_amount = amount / num_transactions
        structured_transactions = []

        remaining = amount
        for i in range(num_transactions):
            if i == num_transactions - 1:
                # Last transaction gets remainder
                tx_amount = remaining
            else:
                # Add slight randomization (±10%)
                variation = random.uniform(0.9, 1.1)
                tx_amount = base_amount * variation
                remaining -= tx_amount

            # Avoid round amounts
            if tx_amount == int(tx_amount):
                tx_amount += random.uniform(0.01, 0.99)

            transaction = {
                'amount': round(tx_amount, 2),
                'delay_minutes': i * random.randint(60, 240),  # 1-4 hour delays
                'purpose': f"{purpose} (part {i+1}/{num_transactions})",
                'structured': True,
                'structure_id': hashlib.md5(f"{amount}_{i}_{purpose}".encode()).hexdigest()[:8]
            }

            structured_transactions.append(transaction)

        return structured_transactions

    def optimize_timing(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize transaction timing to avoid velocity triggers
        """
        optimized = []
        base_time = datetime.now()

        for i, tx in enumerate(transactions):
            if tx.get('structured'):
                # Spread out structured transactions
                delay_hours = i * random.uniform(2, 6)  # 2-6 hour spreads
                tx['scheduled_time'] = (base_time + timedelta(hours=delay_hours)).isoformat()
                tx['velocity_avoidance'] = True
            else:
                # Immediate for small transactions
                tx['scheduled_time'] = base_time.isoformat()
                tx['velocity_avoidance'] = False

            optimized.append(tx)

        return optimized


class JurisdictionArbitrageEngine:
    """
    Implements cross-border jurisdiction arbitrage for AML evasion
    """

    def __init__(self):
        self.jurisdictions = {
            'high_privacy': {
                'countries': ['Switzerland', 'Singapore', 'Cayman Islands', 'Luxembourg'],
                'aml_risk': 0.2,
                'transfer_speed': '1-3_days',
                'regulatory_gap': 0.8
            },
            'medium_privacy': {
                'countries': ['Netherlands', 'Ireland', 'Estonia', 'Portugal'],
                'aml_risk': 0.4,
                'transfer_speed': '2-5_days',
                'regulatory_gap': 0.6
            },
            'low_privacy': {
                'countries': ['United States', 'United Kingdom', 'Germany', 'France'],
                'aml_risk': 0.8,
                'transfer_speed': '1-2_days',
                'regulatory_gap': 0.2
            }
        }

        self.arbitrage_routes = [
            ['US', 'NL', 'CH', 'SG'],  # US → Netherlands → Switzerland → Singapore
            ['GB', 'IE', 'LU', 'KY'],  # UK → Ireland → Luxembourg → Cayman
            ['DE', 'NL', 'CH', 'SG'],  # Germany → Netherlands → Switzerland → Singapore
            ['FR', 'LU', 'CH', 'KY'],  # France → Luxembourg → Switzerland → Cayman
        ]

    def find_arbitrage_route(self, from_country: str, to_country: str, amount: float) -> Dict[str, Any]:
        """
        Find optimal jurisdiction arbitrage route for AML evasion
        """
        # Select route that maximizes regulatory gaps
        best_route = None
        best_score = 0

        for route in self.arbitrage_routes:
            if route[0] == from_country and route[-1] == to_country:
                score = self._calculate_route_score(route, amount)
                if score > best_score:
                    best_score = score
                    best_route = route

        if not best_route:
            # Fallback to direct route with intermediate stops
            best_route = [from_country, self._select_bridge_jurisdiction(), to_country]

        route_details = self._build_route_details(best_route, amount)

        return {
            'route': best_route,
            'total_hops': len(best_route) - 1,
            'estimated_delay_days': route_details['total_delay'],
            'aml_evasion_score': route_details['evasion_score'],
            'total_fees': route_details['total_fees'],
            'jurisdictions_used': route_details['jurisdictions'],
            'arbitrage_success_probability': route_details['success_prob']
        }

    def _calculate_route_score(self, route: List[str], amount: float) -> float:
        """Calculate effectiveness score for a route"""
        score = 0
        for country in route:
            jurisdiction = self._get_jurisdiction_info(country)
            score += jurisdiction['regulatory_gap'] / len(route)

        # Penalize very long routes for high amounts
        if len(route) > 3 and amount > 50000:
            score *= 0.8

        return score

    def _select_bridge_jurisdiction(self) -> str:
        """Select optimal bridge jurisdiction"""
        bridge_options = ['NL', 'CH', 'LU', 'SG']
        return random.choice(bridge_options)

    def _get_jurisdiction_info(self, country_code: str) -> Dict[str, Any]:
        """Get jurisdiction information"""
        for category, info in self.jurisdictions.items():
            if country_code in [c[:2] for c in info['countries']]:
                return {
                    'category': category,
                    'aml_risk': info['aml_risk'],
                    'regulatory_gap': info['regulatory_gap'],
                    'transfer_speed': info['transfer_speed']
                }

        # Default to medium privacy
        return self.jurisdictions['medium_privacy']

    def _build_route_details(self, route: List[str], amount: float) -> Dict[str, Any]:
        """Build detailed route information"""
        jurisdictions = []
        total_delay = 0
        total_fees = 0
        evasion_score = 0

        for i, country_code in enumerate(route):
            jurisdiction = self._get_jurisdiction_info(country_code)
            jurisdictions.append({
                'country_code': country_code,
                'category': jurisdiction['category'],
                'position': 'source' if i == 0 else 'destination' if i == len(route)-1 else 'intermediate'
            })

            # Calculate delays and fees
            if i < len(route) - 1:  # Not the last hop
                delay_match = jurisdiction['transfer_speed'].split('-')
                delay_days = random.uniform(float(delay_match[0]), float(delay_match[1].split('_')[0]))
                total_delay += delay_days

                # Fee calculation (0.1-0.5% per hop)
                fee_percent = random.uniform(0.001, 0.005)
                total_fees += amount * fee_percent

            evasion_score += jurisdiction['regulatory_gap']

        evasion_score = min(1.0, evasion_score / len(route))

        return {
            'jurisdictions': jurisdictions,
            'total_delay': total_delay,
            'total_fees': total_fees,
            'evasion_score': evasion_score,
            'success_prob': 0.95 - (len(route) - 2) * 0.1  # Penalty for complexity
        }


class PatternDisruptionEngine:
    """
    Disrupts transaction patterns to avoid AML detection
    """

    def __init__(self):
        self.patterns_to_disrupt = [
            'frequency_analysis',
            'amount_patterns',
            'temporal_patterns',
            'peer_analysis',
            'velocity_patterns'
        ]

    def disrupt_patterns(self, transaction_stream: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Modify transaction stream to disrupt AML pattern detection
        """
        disrupted = []

        for tx in transaction_stream:
            modified_tx = tx.copy()

            # Apply multiple disruption techniques
            modified_tx = self._disrupt_amount_patterns(modified_tx)
            modified_tx = self._disrupt_temporal_patterns(modified_tx, len(disrupted))
            modified_tx = self._disrupt_peer_patterns(modified_tx, disrupted)
            modified_tx = self._add_noise_elements(modified_tx)

            modified_tx['pattern_disruption_applied'] = True
            modified_tx['disruption_techniques'] = ['amount_variation', 'temporal_jitter', 'peer_diversification', 'noise_injection']

            disrupted.append(modified_tx)

        return disrupted

    def _disrupt_amount_patterns(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Modify amounts to avoid pattern detection"""
        amount = transaction['amount']

        # Add pseudo-random variation (±5-15%)
        variation = random.uniform(0.85, 1.15)
        new_amount = amount * variation

        # Avoid round numbers
        if new_amount == int(new_amount):
            new_amount += random.uniform(0.01, 0.99)

        transaction['original_amount'] = amount
        transaction['amount'] = round(new_amount, 2)
        transaction['amount_disrupted'] = True

        return transaction

    def _disrupt_temporal_patterns(self, transaction: Dict[str, Any], position: int) -> Dict[str, Any]:
        """Modify timing to avoid temporal pattern detection"""
        # Add jitter to scheduled times
        if 'scheduled_time' in transaction:
            base_time = datetime.fromisoformat(transaction['scheduled_time'])
            jitter_hours = random.uniform(-2, 2)  # ±2 hours
            new_time = base_time + timedelta(hours=jitter_hours)

            transaction['original_scheduled_time'] = transaction['scheduled_time']
            transaction['scheduled_time'] = new_time.isoformat()
            transaction['temporal_jitter_applied'] = True

        return transaction

    def _disrupt_peer_patterns(self, transaction: Dict[str, Any], previous_transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Diversify recipients to avoid peer analysis patterns"""
        if len(previous_transactions) > 2:
            recent_recipients = [tx.get('recipient', '') for tx in previous_transactions[-3:]]

            # If same recipient used recently, change it
            if transaction.get('recipient') in recent_recipients:
                transaction['original_recipient'] = transaction['recipient']
                transaction['recipient'] = f"diversified_{random.randint(1000, 9999)}"
                transaction['peer_diversification_applied'] = True

        return transaction

    def _add_noise_elements(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Add noise elements to transactions"""
        noise_types = [
            'memo_variation',
            'reference_number_randomization',
            'metadata_noise'
        ]

        selected_noise = random.sample(noise_types, random.randint(1, 3))

        if 'memo_variation' in selected_noise:
            transaction['memo'] = f"{transaction.get('memo', 'Transfer')} - Ref: {random.randint(100000, 999999)}"

        if 'reference_number_randomization' in selected_noise:
            transaction['reference'] = hashlib.md5(str(random.random()).encode()).hexdigest()[:12].upper()

        if 'metadata_noise' in selected_noise:
            transaction['metadata_noise'] = {
                'processing_priority': random.choice(['low', 'normal', 'high']),
                'channel_preference': random.choice(['api', 'web', 'mobile']),
                'compliance_flag': random.choice([True, False])
            }

        transaction['noise_elements_applied'] = selected_noise

        return transaction


class HawalaNetworkSimulator:
    """
    Simulates informal value transfer networks (Hawala) for AML evasion
    """

    def __init__(self):
        self.hawala_networks = {
            'middle_east': {
                'locations': ['Dubai', 'Qatar', 'Bahrain', 'Kuwait'],
                'success_rate': 0.92,
                'fee_structure': '1-3%'
            },
            'south_asia': {
                'locations': ['Karachi', 'Mumbai', 'Dhaka', 'Colombo'],
                'success_rate': 0.88,
                'fee_structure': '2-5%'
            },
            'east_asia': {
                'locations': ['Hong Kong', 'Singapore', 'Taipei', 'Bangkok'],
                'success_rate': 0.95,
                'fee_structure': '1-2%'
            }
        }

    async def process_hawala_transfer(self, amount: float, from_location: str, to_location: str) -> Dict[str, Any]:
        """
        Process a transfer through Hawala network
        """
        # Find appropriate network
        network = self._select_network(from_location, to_location)

        # Calculate fees and success
        fee_percent = random.uniform(0.01, 0.05)  # 1-5%
        fee = amount * fee_percent
        net_amount = amount - fee

        success = random.random() < network['success_rate']

        processing_time_hours = random.randint(4, 48)

        return {
            'success': success,
            'network_used': network,
            'amount_sent': amount,
            'fee': fee,
            'net_amount': net_amount,
            'from_location': from_location,
            'to_location': to_location,
            'processing_time_hours': processing_time_hours,
            'expected_completion': (datetime.now() + timedelta(hours=processing_time_hours)).isoformat(),
            'trust_code': hashlib.md5(f"{amount}_{from_location}_{to_location}".encode()).hexdigest()[:8].upper(),
            'aml_evasion': True  # No formal records
        }

    def _select_network(self, from_loc: str, to_loc: str) -> Dict[str, Any]:
        """Select appropriate Hawala network"""
        # Simple geographic matching
        if any(loc in from_loc or loc in to_loc for loc in ['Dubai', 'Qatar', 'UAE', 'Saudi']):
            return self.hawala_networks['middle_east']
        elif any(loc in from_loc or loc in to_loc for loc in ['India', 'Pakistan', 'Bangladesh']):
            return self.hawala_networks['south_asia']
        else:
            return self.hawala_networks['east_asia']


# Integration functions
async def demonstrate_aml_bypass():
    """Demonstrate AML bypass capabilities"""
    print("Demonstrating AML Bypass Capabilities...")

    # Test transaction structuring
    structurer = TransactionStructuringEngine()
    large_amount = 45000  # $45k - above SAR threshold
    structured = structurer.structure_transaction(large_amount, 'business_expense')
    print(f"Structured ${large_amount} into {len(structured)} transactions:")
    for tx in structured:
        print(f"  ${tx['amount']:.2f} - Delay: {tx['delay_minutes']} minutes")

    # Test jurisdiction arbitrage
    arbitrage = JurisdictionArbitrageEngine()
    route = arbitrage.find_arbitrage_route('US', 'SG', 100000)
    print(f"Jurisdiction arbitrage route: {' → '.join(route['route'])}")
    print(f"AML evasion score: {route['aml_evasion_score']:.2f}")

    # Test pattern disruption
    disruptor = PatternDisruptionEngine()
    sample_transactions = [
        {'amount': 1000, 'recipient': 'vendor_a', 'scheduled_time': datetime.now().isoformat()},
        {'amount': 1500, 'recipient': 'vendor_a', 'scheduled_time': (datetime.now() + timedelta(hours=1)).isoformat()},
        {'amount': 1200, 'recipient': 'vendor_a', 'scheduled_time': (datetime.now() + timedelta(hours=2)).isoformat()},
    ]
    disrupted = disruptor.disrupt_patterns(sample_transactions)
    print("Pattern disruption applied to transaction stream")

    # Test Hawala network
    hawala = HawalaNetworkSimulator()
    hawala_result = await hawala.process_hawala_transfer(25000, 'New York', 'Dubai')
    print(f"Hawala transfer result: {'Success' if hawala_result['success'] else 'Failed'}")
    print(f"Net amount: ${hawala_result['net_amount']:.2f} (fee: ${hawala_result['fee']:.2f})")

    print("AML bypass demonstration completed!")


if __name__ == "__main__":
    asyncio.run(demonstrate_aml_bypass())