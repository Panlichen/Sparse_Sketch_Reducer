def build(task_name, seed, device, timer, **kwargs):
    if task_name == "Cifar":
        from .cifar import CifarTask

        return CifarTask(
            seed=seed,
            device=device,
            timer=timer,
            architecture=kwargs.get("task_architecture", "ResNet18"),
        )

    elif task_name == "LanguageModeling":
        from .language_modeling import LanguageModelingTask

        return LanguageModelingTask(
            seed=seed, device=device, timer=timer, batch_size=kwargs.get("optimizer_batch_size"), dataset_name=kwargs.get('dataset_name')
        )

    else:
        raise ValueError("Unknown task name")
