"""
Currency conversion service for BRAF Monetization System
Handles USD to NGN conversion for OPay/PalmPay withdrawals with real-time rates
"""

import requests
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import json
import os
import logging

# Setup logging
logger = logging.getLogger(__name__)

class CurrencyConverter:
    """Currency conversion service with real-time rates and caching"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)  # Cache rates for 15 minutes for real-time accuracy
        
        # Updated fallback rates (more current as of Dec 2024)
        self.fallback_rates = {
            "USD_NGN": 1452.0,  # 1 USD = 1452 NGN (current rate)
            "USD_KES": 129.0,   # 1 USD = 129 KES
            "USD_GHS": 15.8,    # 1 USD = 15.8 GHS
            "USD_ZAR": 18.2,    # 1 USD = 18.2 ZAR
            "USD_EGP": 49.5,    # 1 USD = 49.5 EGP
        }
        
        # API endpoints with priorities (most reliable first)
        self.api_endpoints = [
            {
                "name": "ExchangeRate-API",
                "func": self._fetch_from_exchangerate_api,
                "free": True,
                "reliable": True
            },
            {
                "name": "CurrencyAPI",
                "func": self._fetch_from_currencyapi_com,
                "free": True,
                "reliable": True
            },
            {
                "name": "Fixer.io",
                "func": self._fetch_from_fixer_io,
                "free": False,
                "reliable": True
            },
            {
                "name": "CurrencyLayer",
                "func": self._fetch_from_currencylayer,
                "free": False,
                "reliable": True
            },
            {
                "name": "OpenExchangeRates",
                "func": self._fetch_from_openexchangerates,
                "free": False,
                "reliable": True
            }
        ]
    
    def get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        """Get current exchange rate between two currencies"""
        
        if from_currency == to_currency:
            return 1.0
        
        rate_key = f"{from_currency}_{to_currency}"
        
        # Check cache first
        if rate_key in self.cache:
            cached_data = self.cache[rate_key]
            if datetime.utcnow() - cached_data["timestamp"] < self.cache_duration:
                return cached_data["rate"]
        
        # Try to get live rate
        try:
            rate = self._fetch_live_rate(from_currency, to_currency)
            if rate:
                # Cache the rate
                self.cache[rate_key] = {
                    "rate": rate,
                    "timestamp": datetime.utcnow()
                }
                return rate
        except Exception as e:
            print(f"Failed to fetch live rate: {e}")
        
        # Use fallback rate
        if rate_key in self.fallback_rates:
            return self.fallback_rates[rate_key]
        
        # If no fallback available, try reverse rate
        reverse_key = f"{to_currency}_{from_currency}"
        if reverse_key in self.fallback_rates:
            return 1.0 / self.fallback_rates[reverse_key]
        
        raise ValueError(f"No exchange rate available for {from_currency} to {to_currency}")
    
    def _fetch_live_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Fetch live exchange rate from multiple APIs with fallback"""
        
        logger.info(f"Fetching live rate: {from_currency} to {to_currency}")
        
        # Try APIs in order of reliability
        for api_config in self.api_endpoints:
            try:
                logger.debug(f"Trying {api_config['name']} API...")
                rate = api_config["func"](from_currency, to_currency)
                if rate and rate > 0:
                    logger.info(f"Successfully fetched rate from {api_config['name']}: {rate}")
                    return rate
            except Exception as e:
                logger.warning(f"{api_config['name']} API failed: {str(e)}")
                continue
        
        logger.error(f"All APIs failed for {from_currency} to {to_currency}")
        return None
    
    def _fetch_from_exchangerate_api(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Fetch rate from exchangerate-api.com (free tier, very reliable)"""
        try:
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
            
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'BRAF-Monetization-System/1.0'
            })
            
            if response.status_code == 200:
                data = response.json()
                rates = data.get("rates", {})
                rate = rates.get(to_currency)
                if rate:
                    return float(rate)
            
            return None
        except Exception as e:
            logger.error(f"ExchangeRate-API error: {e}")
            return None
    
    def _fetch_from_fixer_io(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Fetch rate from fixer.io (requires API key)"""
        try:
            api_key = os.getenv("FIXER_API_KEY")
            if not api_key:
                return None
            
            url = f"https://data.fixer.io/api/latest?access_key={api_key}&base={from_currency}&symbols={to_currency}"
            
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'BRAF-Monetization-System/1.0'
            })
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    rates = data.get("rates", {})
                    rate = rates.get(to_currency)
                    if rate:
                        return float(rate)
            
            return None
        except Exception as e:
            logger.error(f"Fixer.io error: {e}")
            return None
    
    def _fetch_from_currencylayer(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Fetch rate from currencylayer.com"""
        try:
            api_key = os.getenv("CURRENCYLAYER_API_KEY")
            if not api_key:
                return None
            
            url = f"https://api.currencylayer.com/live?access_key={api_key}&source={from_currency}&currencies={to_currency}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    quotes = data.get("quotes", {})
                    rate_key = f"{from_currency}{to_currency}"
                    rate = quotes.get(rate_key)
                    if rate:
                        return float(rate)
            
            return None
        except Exception as e:
            logger.error(f"CurrencyLayer error: {e}")
            return None
    
    def _fetch_from_openexchangerates(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Fetch rate from openexchangerates.org"""
        try:
            api_key = os.getenv("OPENEXCHANGERATES_API_KEY")
            if not api_key:
                return None
            
            url = f"https://openexchangerates.org/api/latest.json?app_id={api_key}&base={from_currency}&symbols={to_currency}"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                rates = data.get("rates", {})
                rate = rates.get(to_currency)
                if rate:
                    return float(rate)
            
            return None
        except Exception as e:
            logger.error(f"OpenExchangeRates error: {e}")
            return None
    
    def _fetch_from_currencyapi_com(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Fetch rate from currencyapi.com (free tier)"""
        try:
            # Try without API key first (free tier)
            url = f"https://api.currencyapi.com/v3/latest?base_currency={from_currency}&currencies={to_currency}"
            
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'BRAF-Monetization-System/1.0'
            })
            
            if response.status_code == 200:
                data = response.json()
                currencies = data.get("data", {})
                currency_data = currencies.get(to_currency, {})
                rate = currency_data.get("value")
                if rate:
                    return float(rate)
            
            # Try with API key if available
            api_key = os.getenv("CURRENCY_API_KEY")
            if api_key:
                url = f"https://api.currencyapi.com/v3/latest?apikey={api_key}&base_currency={from_currency}&currencies={to_currency}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    currencies = data.get("data", {})
                    currency_data = currencies.get(to_currency, {})
                    rate = currency_data.get("value")
                    if rate:
                        return float(rate)
            
            return None
        except Exception as e:
            logger.error(f"CurrencyAPI.com error: {e}")
            return None
    
    def get_rate_info(self, from_currency: str, to_currency: str) -> Dict:
        """Get detailed rate information including source and freshness"""
        rate_key = f"{from_currency}_{to_currency}"
        
        # Check if we have cached data
        is_cached = False
        cache_age = None
        if rate_key in self.cache:
            cached_data = self.cache[rate_key]
            cache_age = datetime.utcnow() - cached_data["timestamp"]
            is_cached = cache_age < self.cache_duration
        
        rate = self.get_exchange_rate(from_currency, to_currency)
        
        return {
            "from_currency": from_currency,
            "to_currency": to_currency,
            "rate": rate,
            "is_cached": is_cached,
            "cache_age_minutes": cache_age.total_seconds() / 60 if cache_age else None,
            "is_fallback": rate_key in self.fallback_rates and rate == self.fallback_rates[rate_key],
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def convert_amount(self, amount: float, from_currency: str, to_currency: str) -> Dict:
        """Convert amount from one currency to another with detailed info"""
        
        if from_currency == to_currency:
            return {
                "original_amount": amount,
                "original_currency": from_currency,
                "converted_amount": amount,
                "converted_currency": to_currency,
                "exchange_rate": 1.0,
                "conversion_time": datetime.utcnow().isoformat(),
                "rate_source": "no_conversion_needed",
                "is_live_rate": True
            }
        
        rate_info = self.get_rate_info(from_currency, to_currency)
        rate = rate_info["rate"]
        converted_amount = amount * rate
        
        return {
            "original_amount": amount,
            "original_currency": from_currency,
            "converted_amount": round(converted_amount, 2),
            "converted_currency": to_currency,
            "exchange_rate": rate,
            "conversion_time": datetime.utcnow().isoformat(),
            "rate_source": "fallback" if rate_info["is_fallback"] else "live_api",
            "is_live_rate": not rate_info["is_fallback"],
            "is_cached": rate_info["is_cached"],
            "cache_age_minutes": rate_info["cache_age_minutes"]
        }
    
    def get_provider_currency(self, provider: str) -> str:
        """Get the currency used by a payment provider"""
        provider_currencies = {
            "opay": "NGN",
            "palmpay": "NGN",
            "crypto": "USD",  # Crypto typically in USD
            "bank_transfer": "USD"
        }
        
        return provider_currencies.get(provider, "USD")
    
    def calculate_withdrawal_amounts(self, usd_amount: float, provider: str) -> Dict:
        """Calculate withdrawal amounts with currency conversion"""
        
        provider_currency = self.get_provider_currency(provider)
        
        # Convert USD to provider currency
        conversion = self.convert_amount(usd_amount, "USD", provider_currency)
        
        # Calculate fees (in provider currency)
        fee_rates = {
            "opay": 0.015,      # 1.5%
            "palmpay": 0.015,   # 1.5%
            "crypto": 0.01,     # 1.0%
            "bank_transfer": 0.02  # 2.0%
        }
        
        fee_rate = fee_rates.get(provider, 0.02)
        converted_amount = conversion["converted_amount"]
        fee_amount = converted_amount * fee_rate
        net_amount = converted_amount - fee_amount
        
        # Minimum amounts (in provider currency)
        min_amounts = {
            "NGN": 1000,   # 1000 NGN minimum
            "USD": 10,     # $10 minimum
            "KES": 500,    # 500 KES minimum
            "GHS": 50      # 50 GHS minimum
        }
        
        min_amount = min_amounts.get(provider_currency, 10)
        
        return {
            "original_usd_amount": usd_amount,
            "provider_currency": provider_currency,
            "converted_amount": converted_amount,
            "fee_amount": round(fee_amount, 2),
            "net_amount": round(net_amount, 2),
            "exchange_rate": conversion["exchange_rate"],
            "minimum_amount": min_amount,
            "is_valid": net_amount >= min_amount,
            "conversion_details": conversion
        }

# Global currency converter instance
currency_converter = CurrencyConverter()