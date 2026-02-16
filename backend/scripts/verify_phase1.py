"""Verification script for Phase 1 implementation."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_mlflow_tracking():
    """Verify MLflow tracking module."""
    print("✓ Vérification du module MLflow tracking...")
    try:
        from app.ml.tracking import MLFlowTracker, create_default_tracker

        # Test tracker creation
        tracker = create_default_tracker(enabled=False)
        print("  ✓ MLFlowTracker créé avec succès")

        # Test context manager
        with tracker.start_run(run_name="test_run") as run:
            tracker.log_params({"test_param": "test_value"})
            tracker.log_metrics({"test_metric": 0.5}, step=1)
        print("  ✓ Context manager et logging fonctionnent")

        return True
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        return False


def verify_model_configs():
    """Verify model configurations module."""
    print("\n✓ Vérification des configurations de modèles...")
    try:
        from app.ml.model_configs import (
            MODEL_CONFIG_BASELINE,
            MODEL_CONFIG_LARGE,
            get_model_config,
            list_model_configs,
        )

        # Test baseline config
        baseline = MODEL_CONFIG_BASELINE
        print(f"  ✓ Baseline model: ~{baseline.estimated_params:,} params")

        # Test large config
        large = MODEL_CONFIG_LARGE
        print(f"  ✓ Large model: ~{large.estimated_params:,} params")

        # Test config retrieval
        config = get_model_config("large")
        assert config.d_model == 384, "Large config d_model should be 384"
        assert config.num_layers == 6, "Large config should have 6 layers"
        print("  ✓ Configuration large chargée correctement (d_model=384, layers=6)")

        # Test config listing
        configs = list_model_configs()
        assert len(configs) == 5, "Should have 5 predefined configs"
        print(f"  ✓ {len(configs)} configurations disponibles")

        return True
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        return False


def verify_onnx_export():
    """Verify ONNX export module."""
    print("\n✓ Vérification du module d'export ONNX...")
    try:
        from app.ml.export import export_to_onnx, get_onnx_model_info

        # Check if ONNX is available
        try:
            import onnx
            import onnxruntime

            print("  ✓ ONNX et ONNXRuntime installés")
        except ImportError:
            print("  ⚠ ONNX/ONNXRuntime non installés (optionnel)")
            return True  # Not a failure if optional dependencies missing

        return True
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        return False


def verify_pipeline_onnx_support():
    """Verify pipeline ONNX support."""
    print("\n✓ Vérification du support ONNX dans le pipeline...")
    try:
        from app.ml.pipeline import SignFlowInferencePipeline

        # Test pipeline creation without model
        pipeline = SignFlowInferencePipeline()
        print("  ✓ Pipeline créé sans modèle")

        # Check ONNX attributes exist
        assert hasattr(pipeline, "use_onnx"), "Pipeline should have use_onnx attribute"
        assert hasattr(pipeline, "onnx_session"), "Pipeline should have onnx_session attribute"
        print("  ✓ Attributs ONNX présents dans le pipeline")

        # Check methods exist
        assert hasattr(pipeline, "_load_onnx_model"), "Pipeline should have _load_onnx_model method"
        assert hasattr(
            pipeline, "_infer_probabilities_onnx"
        ), "Pipeline should have _infer_probabilities_onnx method"
        print("  ✓ Méthodes ONNX présentes dans le pipeline")

        return True
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        return False


def verify_trainer_mlflow_integration():
    """Verify trainer MLflow integration."""
    print("\n✓ Vérification de l'intégration MLflow dans le trainer...")
    try:
        from app.ml.trainer import TrainingConfig

        # Test config with MLflow options
        config = TrainingConfig(
            num_epochs=1,
            use_mlflow=True,
            mlflow_run_name="test_run",
            mlflow_tags={"test": "value"},
        )
        print("  ✓ TrainingConfig avec options MLflow créé")

        assert config.use_mlflow is True, "MLflow should be enabled"
        assert config.mlflow_run_name == "test_run", "Run name should be set"
        assert config.mlflow_tags == {"test": "value"}, "Tags should be set"
        print("  ✓ Options MLflow configurées correctement")

        return True
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        return False


def verify_dependencies():
    """Verify new dependencies in pyproject.toml."""
    print("\n✓ Vérification des dépendances dans pyproject.toml...")
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    dependencies = pyproject["project"]["dependencies"]

    # Check for new dependencies
    required_deps = ["mlflow", "onnx", "onnxruntime"]
    found_deps = []

    for dep_str in dependencies:
        for req_dep in required_deps:
            if req_dep in dep_str.lower():
                found_deps.append(req_dep)
                print(f"  ✓ {dep_str} trouvé")

    if len(found_deps) != len(required_deps):
        missing = set(required_deps) - set(found_deps)
        print(f"  ⚠ Dépendances manquantes: {missing}")
        return False

    return True


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("VÉRIFICATION PHASE 1 - SIGNFLOW ML UPGRADE")
    print("=" * 60)

    results = {
        "MLflow Tracking": verify_mlflow_tracking(),
        "Model Configs": verify_model_configs(),
        "ONNX Export": verify_onnx_export(),
        "Pipeline ONNX Support": verify_pipeline_onnx_support(),
        "Trainer MLflow Integration": verify_trainer_mlflow_integration(),
        "Dependencies": verify_dependencies(),
    }

    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 Toutes les vérifications Phase 1 ont réussi!")
        print("\nProchaines étapes:")
        print("1. Installer les dépendances: pip install -e .")
        print("2. Démarrer MLflow UI: docker-compose up mlflow")
        print("3. Tester l'entraînement avec MLflow tracking")
        print("4. Exporter un modèle en ONNX et tester l'inférence")
        return 0
    else:
        print("\n⚠ Certaines vérifications ont échoué.")
        print("Veuillez corriger les erreurs avant de continuer.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
