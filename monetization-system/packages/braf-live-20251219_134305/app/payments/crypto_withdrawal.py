from web3 import Web3
import os
from typing import Dict
from decimal import Decimal

class CryptoWithdrawal:
    def __init__(self):
        # Initialize blockchain connections
        self.networks = {
            "TRC20": {
                "rpc": "https://api.trongrid.io",
                "usdt_contract": "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t",
                "decimals": 6
            },
            "ERC20": {
                "rpc": "https://mainnet.infura.io/v3/{INFURA_KEY}",
                "usdt_contract": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                "decimals": 6
            },
            "BTC": {
                "network": "mainnet",
                "fee_rate": 50  # satoshis per byte
            },
            "ETH": {
                "rpc": "https://mainnet.infura.io/v3/{INFURA_KEY}",
                "gas_price": Web3.to_wei('50', 'gwei')
            }
        }

    async def process_withdrawal(self, enterprise_id: str, amount: float,
                                cryptocurrency: str, network: str,
                                wallet_address: str) -> Dict:
        """Process cryptocurrency withdrawal"""
        
        # Validate wallet address
        if not self.validate_address(cryptocurrency, wallet_address):
            raise ValueError("Invalid wallet address")
        
        # Check blockchain confirmation requirement
        if cryptocurrency != "USDT":
            # For BTC/ETH, require additional confirmations
            required_confirmations = 3
        else:
            required_confirmations = 12  # USDT requires more confirmations
        
        # Process based on cryptocurrency
        if cryptocurrency == "USDT":
            tx_hash = await self.send_usdt(
                amount=amount,
                to_address=wallet_address,
                network=network
            )
        elif cryptocurrency == "BTC":
            tx_hash = await self.send_btc(
                amount=amount,
                to_address=wallet_address
            )
        elif cryptocurrency == "ETH":
            tx_hash = await self.send_eth(
                amount=amount,
                to_address=wallet_address
            )
        else:
            raise ValueError(f"Unsupported cryptocurrency: {cryptocurrency}")
        
        # Wait for confirmations
        confirmations = await self.wait_for_confirmations(
            tx_hash=tx_hash,
            cryptocurrency=cryptocurrency,
            required=required_confirmations
        )
        
        return {
            "transaction_hash": tx_hash,
            "amount": amount,
            "currency": cryptocurrency,
            "network": network,
            "recipient": wallet_address,
            "confirmations": confirmations,
            "status": "completed" if confirmations >= required_confirmations else "pending",
            "timestamp": datetime.utcnow().isoformat()
        }

    async def send_usdt(self, amount: float, to_address: str, network: str) -> str:
        """Send USDT via specified network"""
        config = self.networks[network]
        
        if network == "TRC20":
            # Tron network implementation
            from tronpy import Tron
            client = Tron()
            
            # Convert amount to sun (1 USDT = 1,000,000 sun)
            amount_sun = int(amount * 10**6)
            
            # Build transaction
            txn = (
                client.trx.transfer(
                    config["usdt_contract"],
                    to_address,
                    amount_sun
                )
                .memo("BRAF Earnings Withdrawal")
                .build()
                .sign(private_key=os.getenv("TRON_PRIVATE_KEY"))
            )
            
            result = txn.broadcast()
            return result['txid']
        
        elif network == "ERC20":
            # Ethereum network implementation
            w3 = Web3(Web3.HTTPProvider(config["rpc"]))
            
            # Load contract
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(config["usdt_contract"]),
                abi=self.load_abi("usdt_erc20")
            )
            
            # Build transaction
            nonce = w3.eth.get_transaction_count(os.getenv("ETH_WALLET_ADDRESS"))
            
            txn = contract.functions.transfer(
                Web3.to_checksum_address(to_address),
                int(amount * 10**6)
            ).build_transaction({
                'chainId': 1,
                'gas': 100000,
                'gasPrice': w3.to_wei('50', 'gwei'),
                'nonce': nonce,
            })
            
            # Sign and send
            signed_txn = w3.eth.account.sign_transaction(
                txn, 
                private_key=os.getenv("ETH_PRIVATE_KEY")
            )
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            return tx_hash.hex()
