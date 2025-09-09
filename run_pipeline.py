#!/usr/bin/env python3
"""
Clinical Crystal Ball: Complete Pipeline Execution
Run all phases of the ML pipeline sequentially
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_execution.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class PipelineExecutor:
    """
    Orchestrates the complete Clinical Crystal Ball pipeline execution
    """

    def __init__(self):
        self.phases = [
            {
                'name': 'Phase 1: Data Generation',
                'script': '01_generate_data.py',
                'description': 'Generate synthetic clinical dataset',
                'outputs': ['patient_static_data.csv', 'synthetic_patient_data.csv']
            },
            {
                'name': 'Phase 2: Data Preprocessing',
                'script': '02_preprocess_data.py', 
                'description': 'Preprocess and engineer features',
                'outputs': ['processed_patient_data.csv']
            },
            {
                'name': 'Phase 3: Model Training',
                'script': '03_train_tft_model.py',
                'description': 'Train TFT model',
                'outputs': ['tft_model.pkl', 'model_predictions.csv']
            },
            {
                'name': 'Phase 4: Model Evaluation',
                'script': '04_evaluate_model.py',
                'description': 'Evaluate model and generate explanations',
                'outputs': ['model_evaluation_plots.png', 'evaluation_results.json']
            }
        ]

        self.start_time = datetime.now()

    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        logger.info("Checking dependencies...")

        required_packages = {
            'pandas': 'pandas',
            'numpy': 'numpy',
            'scikit-learn': 'sklearn',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'plotly': 'plotly',
            'streamlit': 'streamlit',
            'joblib': 'joblib'
        }

        missing_packages = []

        for package_name, module_name in required_packages.items():
            try:
                __import__(module_name)
                logger.info(f"[OK] {package_name}")
            except ImportError:
                missing_packages.append(package_name)
                logger.error(f"[MISSING] {package_name}")

        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.error("Please run: pip install -r requirements.txt")
            return False

        logger.info("All dependencies satisfied")
        return True

    def run_phase(self, phase):
        """Execute a single phase of the pipeline"""
        logger.info(f"Starting {phase['name']}")
        logger.info(f"{phase['description']}")

        try:
            # Check if script exists
            if not os.path.exists(phase['script']):
                logger.error(f"Script not found: {phase['script']}")
                return False

            # Execute the phase script
            result = subprocess.run([sys.executable, phase['script']], 
                                  capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"{phase['name']} completed successfully")

                # Check if expected outputs were created
                missing_outputs = []
                for output_file in phase['outputs']:
                    if not os.path.exists(output_file):
                        missing_outputs.append(output_file)

                if missing_outputs:
                    logger.warning(f"Some expected outputs not found: {missing_outputs}")
                else:
                    logger.info(f"All expected outputs created: {phase['outputs']}")

                return True
            else:
                logger.error(f"{phase['name']} failed")
                logger.error(f"Error: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Exception during {phase['name']}: {e}")
            return False

    def run_complete_pipeline(self):
        """Execute the complete pipeline"""
        logger.info("Starting Clinical Crystal Ball Pipeline")
        logger.info("=" * 60)

        # Check dependencies
        if not self.check_dependencies():
            logger.error("Dependency check failed. Aborting pipeline.")
            return False

        # Execute all phases
        all_phases_successful = True

        for i, phase in enumerate(self.phases, 1):
            logger.info(f"\n{'='*20} PHASE {i} {'='*20}")

            phase_success = self.run_phase(phase)

            if not phase_success:
                logger.error(f"Pipeline failed at {phase['name']}")
                all_phases_successful = False
                break

            logger.info(f"Phase {i} completed successfully")

        # Final status
        if all_phases_successful:
            logger.info("\n" + "*" * 20)
            logger.info("PIPELINE EXECUTION SUCCESSFUL!")
            logger.info("*" * 20)
            logger.info("\nAll phases completed successfully")
            logger.info("\nReady to launch dashboard:")
            logger.info("   streamlit run 05_streamlit_dashboard.py")
        else:
            logger.error("\nPIPELINE EXECUTION FAILED")
            logger.error("Check the logs above for specific error details")

        return all_phases_successful

def main():
    """Main execution function"""
    print("Clinical Crystal Ball - Complete ML Pipeline")
    print("=" * 50)

    executor = PipelineExecutor()
    success = executor.run_complete_pipeline()

    if success:
        print("\nPipeline completed successfully!")
        print("Next step: Launch the dashboard with 'streamlit run 05_streamlit_dashboard.py'")
        sys.exit(0)
    else:
        print("\nPipeline execution failed!")
        print("Check pipeline_execution.log for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()
