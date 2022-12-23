import numpy as np
import tsfel

def featureExtraction(activation, postActivation):
    features = np.array([tsfel.feature_extraction.features.auc(abs(activation), 1000),
    tsfel.feature_extraction.features.calc_mean(abs(activation)),
    tsfel.feature_extraction.features.mean_abs_diff(activation),
    tsfel.feature_extraction.features.auc((activation ** 2), 1000),
    tsfel.feature_extraction.features.auc(abs(postActivation), 1000),
    tsfel.feature_extraction.features.calc_mean(abs(postActivation)),
    tsfel.feature_extraction.features.mean_abs_diff(abs(postActivation)),
    tsfel.feature_extraction.features.auc((postActivation ** 2), 1000),
    tsfel.feature_extraction.features.calc_var(postActivation),
    tsfel.feature_extraction.features.rms(postActivation),
    tsfel.feature_extraction.features.median_frequency(postActivation, 1000)])
    return features