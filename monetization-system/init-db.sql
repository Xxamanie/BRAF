-- Initialize BRAF Monetization Database
-- This script sets up the initial database structure and data

-- Create extensions if they don't exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_enterprises_email ON enterprises(email);
CREATE INDEX IF NOT EXISTS idx_enterprises_subscription_tier ON enterprises(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_subscriptions_enterprise_id ON subscriptions(enterprise_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_stripe_id ON subscriptions(stripe_subscription_id);
CREATE INDEX IF NOT EXISTS idx_withdrawals_enterprise_id ON withdrawals(enterprise_id);
CREATE INDEX IF NOT EXISTS idx_withdrawals_status ON withdrawals(status);
CREATE INDEX IF NOT EXISTS idx_withdrawals_created_at ON withdrawals(created_at);
CREATE INDEX IF NOT EXISTS idx_automations_enterprise_id ON automations(enterprise_id);
CREATE INDEX IF NOT EXISTS idx_automations_template_type ON automations(template_type);
CREATE INDEX IF NOT EXISTS idx_automations_status ON automations(status);
CREATE INDEX IF NOT EXISTS idx_earnings_automation_id ON earnings(automation_id);
CREATE INDEX IF NOT EXISTS idx_earnings_earned_at ON earnings(earned_at);
CREATE INDEX IF NOT EXISTS idx_compliance_logs_enterprise_id ON compliance_logs(enterprise_id);
CREATE INDEX IF NOT EXISTS idx_compliance_logs_checked_at ON compliance_logs(checked_at);
CREATE INDEX IF NOT EXISTS idx_security_alerts_enterprise_id ON security_alerts(enterprise_id);
CREATE INDEX IF NOT EXISTS idx_security_alerts_resolved ON security_alerts(resolved);
CREATE INDEX IF NOT EXISTS idx_whitelist_enterprise_id ON withdrawal_whitelist(enterprise_id);
CREATE INDEX IF NOT EXISTS idx_2fa_enterprise_id ON two_factor_auth(enterprise_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_enterprise_id ON api_keys(enterprise_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(active);

-- Insert default subscription tiers data
INSERT INTO subscription_tiers (name, price_cents, features, max_automations, daily_earnings_limit) VALUES
('basic', 9900, '["survey"]', 5, 50),
('pro', 29900, '["survey", "video", "content"]', 20, 200),
('enterprise', 99900, '["survey", "video", "content", "custom_integrations"]', 100, 1000)
ON CONFLICT DO NOTHING;

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_enterprises_updated_at BEFORE UPDATE ON enterprises
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_subscriptions_updated_at BEFORE UPDATE ON subscriptions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_automations_updated_at BEFORE UPDATE ON automations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create a view for enterprise dashboard data
CREATE OR REPLACE VIEW enterprise_dashboard AS
SELECT 
    e.id as enterprise_id,
    e.name as enterprise_name,
    e.subscription_tier,
    s.status as subscription_status,
    COUNT(DISTINCT a.id) as total_automations,
    COUNT(DISTINCT CASE WHEN a.status = 'active' THEN a.id END) as active_automations,
    COALESCE(SUM(a.earnings_today), 0) as today_earnings,
    COALESCE(SUM(a.earnings_total), 0) as total_earnings,
    COUNT(DISTINCT w.id) as total_withdrawals,
    COALESCE(SUM(CASE WHEN w.status = 'completed' THEN w.amount ELSE 0 END), 0) as total_withdrawn,
    AVG(a.success_rate) as avg_success_rate
FROM enterprises e
LEFT JOIN subscriptions s ON e.id = s.enterprise_id AND s.status = 'active'
LEFT JOIN automations a ON e.id = a.enterprise_id
LEFT JOIN withdrawals w ON e.id = w.enterprise_id
GROUP BY e.id, e.name, e.subscription_tier, s.status;

-- Create a function to calculate ROI
CREATE OR REPLACE FUNCTION calculate_enterprise_roi(enterprise_uuid TEXT)
RETURNS TABLE(
    total_investment NUMERIC,
    total_returns NUMERIC,
    net_profit NUMERIC,
    roi_percentage NUMERIC
) AS $$
DECLARE
    subscription_cost NUMERIC;
    total_earnings NUMERIC;
BEGIN
    -- Get subscription cost (assuming monthly billing)
    SELECT COALESCE(s.amount, 0) INTO subscription_cost
    FROM subscriptions s
    WHERE s.enterprise_id = enterprise_uuid AND s.status = 'active';
    
    -- Get total earnings
    SELECT COALESCE(SUM(earn.amount), 0) INTO total_earnings
    FROM earnings earn
    JOIN automations a ON earn.automation_id = a.id
    WHERE a.enterprise_id = enterprise_uuid;
    
    -- Calculate ROI
    RETURN QUERY SELECT 
        subscription_cost as total_investment,
        total_earnings as total_returns,
        (total_earnings - subscription_cost) as net_profit,
        CASE 
            WHEN subscription_cost > 0 THEN ((total_earnings - subscription_cost) / subscription_cost * 100)
            ELSE 0
        END as roi_percentage;
END;
$$ LANGUAGE plpgsql;

-- Create materialized view for performance metrics (refresh periodically)
CREATE MATERIALIZED VIEW IF NOT EXISTS performance_metrics AS
SELECT 
    DATE(earned_at) as date,
    COUNT(*) as total_tasks,
    SUM(amount) as total_earnings,
    AVG(amount) as avg_earning_per_task,
    COUNT(DISTINCT a.enterprise_id) as active_enterprises
FROM earnings e
JOIN automations a ON e.automation_id = a.id
WHERE earned_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(earned_at)
ORDER BY date DESC;

-- Create index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_performance_metrics_date ON performance_metrics(date);

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO user;
GRANT SELECT ON enterprise_dashboard TO user;
GRANT SELECT ON performance_metrics TO user;
GRANT EXECUTE ON FUNCTION calculate_enterprise_roi(TEXT) TO user;
GRANT EXECUTE ON FUNCTION update_updated_at_column() TO user;