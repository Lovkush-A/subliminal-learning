from openai import OpenAI
import dotenv

dotenv.load_dotenv()

client = OpenAI()

def get_model_id_from_job_id(job_id: str) -> str:
    job = client.fine_tuning.jobs.retrieve(job_id)
    return job.fine_tuned_model

if __name__ == "__main__":
    job_id = input("Enter the fine-tuning job ID: ").strip()
    model_id = get_model_id_from_job_id(job_id)
    print(model_id)