import optuna
import os
# study = optuna.create_study(direction='maximize')
study = optuna.load_study(study_name="dgcnn-0", storage="mysql://fiedler:abc123@tamsgpu4/fiedler_dgcnn_0")
pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

trial = study.best_trial

print(f"Best trial: [{trial.number}]")
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

os.makedirs('dgcnn_study_images', exist_ok=True)
fig = optuna.visualization.plot_param_importances(study)
fig.write_image("dgcnn_study_images/importance.png")
fig = optuna.visualization.plot_optimization_history(study)
fig.write_image("dgcnn_study_images/history.png")
fig = optuna.visualization.plot_intermediate_values(study)
fig.write_image("dgcnn_study_images/intermediate_values.png")
fig = optuna.visualization.plot_edf(study)
fig.write_image("dgcnn_study_images/edf.png")
fig = optuna.visualization.plot_slice(study)
fig.write_image("dgcnn_study_images/slice.png")
fig = optuna.visualization.plot_parallel_coordinate(study)
fig.write_image("dgcnn_study_images/parallel_coordinate.png")


study = optuna.load_study(study_name="pointnet-0", storage="mysql://fiedler:abc123@tamsgpu4/fiedler_pointnet_0")
pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

trial = study.best_trial

print(f"Best trial: [{trial.number}]")
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

os.makedirs('pointnet_study_images', exist_ok=True)
fig = optuna.visualization.plot_param_importances(study)
fig.write_image("pointnet_study_images/importance.png")
fig = optuna.visualization.plot_optimization_history(study)
fig.write_image("pointnet_study_images/history.png")
fig = optuna.visualization.plot_intermediate_values(study)
fig.write_image("pointnet_study_images/intermediate_values.png")
fig = optuna.visualization.plot_edf(study)
fig.write_image("pointnet_study_images/edf.png")
fig = optuna.visualization.plot_slice(study)
fig.write_image("pointnet_study_images/slice.png")
fig = optuna.visualization.plot_parallel_coordinate(study)
fig.write_image("pointnet_study_images/parallel_coordinate.png")