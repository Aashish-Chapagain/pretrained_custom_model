from deepeval.benchmarks import HellaSwag
from deepeval.benchmarks.tasks import HellaSwagTask
import os 

model = os.path.join(os.getcwd(), "final_model.pth")  # Path to your trained model
# Define benchmark with specific tasks and shots
benchmark = HellaSwag(
    tasks=[HellaSwagTask.TRIMMING_BRANCHES_OR_HEDGES, HellaSwagTask.BATON_TWIRLING],
    n_shots=5
)
# Replace 'mistral_7b' with your own custom model
benchmark.evaluate(model=model)
print(benchmark.overall_score)