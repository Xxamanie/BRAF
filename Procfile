web: gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 monetization-system.wsgi:app
release: python monetization-system/database/init_db.py