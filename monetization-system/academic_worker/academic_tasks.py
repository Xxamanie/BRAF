
import os
import time
from celery import Celery

app = Celery('academic_research_tasks')
app.conf.broker_url = os.environ.get('ACADEMIC_REDIS_URL', 'redis://localhost:6379/0')

@app.task
def process_academic_data(data):
    print(f"Processing academic data: {data}")
    time.sleep(2)  # Simulate processing
    return {"status": "processed", "data": data}

if __name__ == '__main__':
    app.start()
