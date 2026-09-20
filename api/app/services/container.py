from api.app.services.comparison_service import ComparisonService
from api.app.services.model_registry import ModelRegistry
from api.app.services.report_service import ReportService
from api.app.services.training_service import TrainingService


model_registry = ModelRegistry()
training_service = TrainingService()
comparison_service = ComparisonService(model_registry)
report_service = ReportService(model_registry)
