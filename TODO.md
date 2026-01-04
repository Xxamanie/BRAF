# Project Reorganization Plan

## Objective
Reorganize the scattered project structure to link important files together, ensuring components like automation, monetization, infrastructure, intelligence, and security are cohesive and no useful parts are idle.

## Approved Plan
Consolidate similar functionalities into a unified structure under monetization-system/ as the main system, grouping by categories: Core, Automation, Monetization, Infrastructure, Intelligence, and Security.

## Steps to Complete

### 1. Merge BRAF/automation into monetization-system/automation
- [ ] Move BRAF/automation/browser_automation.py to monetization-system/automation/
- [ ] Update any imports in the moved file to reference monetization-system paths
- [ ] Remove BRAF/automation/ directory after merge

### 2. Merge BRAF/monetization into monetization-system/earnings
- [ ] Move BRAF/monetization/earnings_tracker.py to monetization-system/earnings/
- [ ] Rename to earnings_tracker_braf.py or integrate with existing swagbucks_integration.py
- [ ] Update imports and dependencies

### 3. Merge BRAF/core into monetization-system/core
- [ ] Move BRAF/core/runner.py to monetization-system/core/
- [ ] Integrate with existing analytics_engine.py if needed
- [ ] Update any references

### 4. Merge BRAF/scrapers into monetization-system/scrapers
- [ ] Move BRAF/scrapers/browser_scraper.py to monetization-system/scrapers/
- [ ] Ensure compatibility with existing browser_scraper.py

### 5. Merge BRAF/payments into monetization-system/payments
- [x] Move BRAF/payments/maxel_integration.py to monetization-system/payments/
- [ ] Integrate with existing crypto_withdrawal.py

### 6. Merge BRAF/workflows into monetization-system/coordination
- [x] Move BRAF/workflows/task_scheduler.py to monetization-system/coordination/
- [ ] Rename or integrate with academic_coordination_engine.py

### 7. Merge BRAF/dashboard into monetization-system/dashboard
- [ ] Move BRAF/dashboard/index.html to monetization-system/templates/ or static/
- [ ] Integrate with existing dashboard_service.py

### 8. Integrate src/braf/ components
- [ ] Review src/braf/ structure and merge relevant parts into monetization-system/
- [ ] Update CLI and core components to reference consolidated paths

### 9. Update all import statements and dependencies
- [x] Search for and update any imports referencing BRAF/ paths
- [x] Ensure all files use relative or absolute paths to monetization-system/

### 10. Remove redundant directories
- [ ] Delete BRAF/ directory after all merges are complete
- [ ] Clean up any empty directories

### 11. Test the reorganized structure
- [ ] Run basic tests to ensure no broken imports
- [ ] Verify key functionalities still work

### 12. Final cleanup
- [ ] Update any documentation or README files
- [ ] Ensure the project runs without errors
