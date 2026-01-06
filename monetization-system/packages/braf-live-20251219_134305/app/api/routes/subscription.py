from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from config import Config
from enterprise.subscription_service import EnterpriseSubscription

router = APIRouter(prefix="/api/v1/enterprise")

class SubscriptionRequest(BaseModel):
    enterprise_id: str
    tier: str
    payment_method_id: str

@router.post("/subscribe")
async def create_subscription(request: SubscriptionRequest):
    """API endpoint for enterprise subscription"""
    try:
        subscription_service = EnterpriseSubscription(
            stripe_api_key=Config.STRIPE_SECRET_KEY
        )
        
        result = subscription_service.create_subscription(
            enterprise_id=request.enterprise_id,
            tier=request.tier,
            payment_method_id=request.payment_method_id
        )
        
        # TODO: Store subscription in database
        # db.save_subscription({
        #     "enterprise_id": request.enterprise_id,
        #     "subscription_data": result,
        #     "created_at": datetime.utcnow(),
        #     "status": "active"
        # })
        
        return {
            "success": True,
            "data": result,
            "message": "Subscription created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
