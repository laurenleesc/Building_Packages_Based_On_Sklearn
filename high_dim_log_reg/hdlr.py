import numpy as np
import pandas
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression


class HDLR(BaseEstimator):
        def __init__(self,
                 feature_names=None,
                 regression=False,
                 precision_min=0.5,
                 recall_min=0.01,
                 n_estimators=10,
                 max_samples=.8,
                 max_samples_features=1.,
                 bootstrap=False,
                 bootstrap_features=False,
                 max_depth=3,
                 max_depth_duplication=None,
                 max_features=1.,
                 min_samples_split=2,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
            self.precision_min = precision_min
            self.recall_min = recall_min
            self.feature_names = feature_names
            self.regression = regression
            self.n_estimators = n_estimators
            self.max_samples = max_samples
            self.max_samples_features = max_samples_features
            self.bootstrap = bootstrap
            self.bootstrap_features = bootstrap_features
            self.max_depth = max_depth
            self.max_depth_duplication = max_depth_duplication
            self.max_features = max_features
            self.min_samples_split = min_samples_split
            self.n_jobs = n_jobs
            self.random_state = random_state
            self.verbose = verbose
    
        def __init__(self,
                 feature_names=None,
                 regression=False,
                 precision_min=0.5,
                 recall_min=0.01,
                 n_estimators=10,
                 max_samples=.8,
                 max_samples_features=1.,
                 bootstrap=False,
                 bootstrap_features=False,
                 max_depth=3,
                 max_depth_duplication=None,
                 max_features=1.,
                 min_samples_split=2,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
            self.precision_min = precision_min
            self.recall_min = recall_min
            self.feature_names = feature_names
            self.regression = regression
            self.n_estimators = n_estimators
            self.max_samples = max_samples
            self.max_samples_features = max_samples_features
            self.bootstrap = bootstrap
            self.bootstrap_features = bootstrap_features
            self.max_depth = max_depth
            self.max_depth_duplication = max_depth_duplication
            self.max_features = max_features
            self.min_samples_split = min_samples_split
            self.n_jobs = n_jobs
            self.random_state = random_state
            self.verbose = verbose

        def fit(self, X, y, sample_weight=None): 

            X, y = check_X_y(X, y)
            check_classification_targets(y)
            self.n_features_ = X.shape[1]

            if not isinstance(self.max_depth_duplication, int) \
                    and self.max_depth_duplication is not None:
                raise ValueError(
                    "max_depth_duplication should be an integer"
                )

            if self.regression:
                self.classes_ = None
                self.means = y.mean(axis=0)
            else:
                self.classes_ = np.unique(y)
                n_classes = len(self.classes_)

                if n_classes < 2:
                    raise ValueError(
                        "This method needs samples of at least 2 classes"
                        " in the data, but the data contains only one"
                        " class: %r" % self.classes_[0]
                    )

                if not set(self.classes_) == set([0, 1]):
                    warn(
                        "Found labels %s. This method assumes target class to be"
                        " labeled as 1 and normal data to be labeled as 0. Any"
                        " label different from 0 will be considered as being from"
                        " the target class." % set(self.classes_)
                    )
                    y = (y > 0)

            # ensure that max_samples is in [1, n_samples]:
            n_samples = X.shape[0]

            if isinstance(self.max_samples, six.string_types):
                raise ValueError('max_samples (%s) is not supported.'
                                'Valid choices are: "auto", int or'
                                'float' % self.max_samples)

            if isinstance(self.max_samples, INTEGER_TYPES):
                if self.max_samples > n_samples:
                    warn("max_samples (%s) is greater than the "
                        "total number of samples (%s). max_samples "
                        "will be set to n_samples for estimation."
                        % (self.max_samples, n_samples))
                    max_samples = n_samples
                else:
                    max_samples = self.max_samples
            else:  # float
                if not (0. < self.max_samples <= 1.):
                    raise ValueError("max_samples must be in (0, 1], got %r"
                                    % self.max_samples)
                max_samples = int(self.max_samples * X.shape[0])

            self.max_samples_ = max_samples

            self.rules_ = {}
            self.estimators_ = []
            self.estimators_samples_ = []
            self.estimators_features_ = []

            # default columns names :
            feature_names_ = [BASE_FEATURE_NAME + x for x in
                            np.arange(X.shape[1]).astype(str)]
            if self.feature_names is not None:
                self.feature_dict_ = {BASE_FEATURE_NAME + str(i): feat
                                    for i, feat in enumerate(self.feature_names)}
            else:
                self.feature_dict_ = {BASE_FEATURE_NAME + str(i): feat
                                    for i, feat in enumerate(feature_names_)}
            self.feature_names_ = feature_names_

            clfs = []
            regs = []

            self._max_depths = self.max_depth \
                if isinstance(self.max_depth, Iterable) else [self.max_depth]

            for max_depth in self._max_depths:
                if not self.regression:
                    # do classification models;
                    # otherwise, just leave classification models as an empty list
                    bagging_clf = BaggingClassifier(
                        base_estimator=DecisionTreeClassifier(
                            max_depth=max_depth,
                            max_features=self.max_features,
                            min_samples_split=self.min_samples_split),
                        n_estimators=self.n_estimators,
                        max_samples=self.max_samples_,
                        max_features=self.max_samples_features,
                        bootstrap=self.bootstrap,
                        bootstrap_features=self.bootstrap_features,
                        # oob_score=... XXX may be added
                        # if selection on tree perf needed.
                        # warm_start=... XXX may be added to
                        # increase computation perf.
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        verbose=self.verbose)
                    clfs.append(bagging_clf)

                bagging_reg = BaggingRegressor(
                    base_estimator=DecisionTreeRegressor(
                        max_depth=max_depth,
                        max_features=self.max_features,
                        min_samples_split=self.min_samples_split),
                    n_estimators=self.n_estimators,
                    max_samples=self.max_samples_,
                    max_features=self.max_samples_features,
                    bootstrap=self.bootstrap,
                    bootstrap_features=self.bootstrap_features,
                    # oob_score=... XXX may be added
                    # if selection on tree perf needed.
                    # warm_start=... XXX may be added to increase computation perf.
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbose=self.verbose)

                regs.append(bagging_reg)

            # define regression target:
            if sample_weight is not None and not self.regression:
                if sample_weight is not None:
                    sample_weight = check_array(sample_weight, ensure_2d=False)
                weights = sample_weight - sample_weight.min()
                contamination = float(sum(y)) / len(y)
                y_reg = (
                    pow(weights, 0.5) * 0.5 / contamination * (y > 0) -
                    pow((weights).mean(), 0.5) * (y == 0))
                y_reg = 1. / (1 + np.exp(-y_reg))  # sigmoid
            else:
                y_reg = y  # same as an other classification bagging

            for clf in clfs:
                clf.fit(X, y)
                self.estimators_ += clf.estimators_
                self.estimators_samples_ += clf.estimators_samples_
                self.estimators_features_ += clf.estimators_features_

            for reg in regs:
                reg.fit(X, y_reg)
                self.estimators_ += reg.estimators_
                self.estimators_samples_ += reg.estimators_samples_
                self.estimators_features_ += reg.estimators_features_

            rules_ = []
            for estimator, samples, features in zip(self.estimators_,
                                                    self.estimators_samples_,
                                                    self.estimators_features_):

                if not self.regression:
                    # Create mask for OOB samples
                    mask = ~samples
                    if sum(mask) == 0:
                        warn("OOB evaluation not possible: doing it in-bag."
                            " Performance evaluation is likely to be wrong"
                            " (overfitting) and selected rules are likely to"
                            " not perform well! Please use max_samples < 1.")
                        mask = samples
                    rules_from_tree = self._tree_to_rules(
                        estimator, np.array(self.feature_names_)[features]
                    )
                    # XXX todo: idem without dataframe
                    X_oob = pandas.DataFrame((X[mask, :])[:, features],
                                            columns=np.array(
                                                self.feature_names_)[features])
                    y_mask = y[mask]

                else:  # self.regression:
                    X_oob = pandas.DataFrame(
                        (X)[:, features],
                        columns=np.array(
                            self.feature_names_)[features]
                    )
                    y_mask = y
                    rules_from_tree = self._tree_to_rules(
                        estimator, np.array(self.feature_names_)[features]
                    )

                if X_oob.shape[1] > 1:  # otherwise pandas bug (cf. issue #16363)
                    y_oob = y_mask
                    if not self.regression:
                        y_oob = np.array((y_oob != 0))

                    # Add OOB performances to rules:
                    rules_from_tree = [(r, self._eval_rule_perf(*r, X_oob, y_oob))
                                    for r in set(rules_from_tree)]
                    rules_ += rules_from_tree

            # Factorize rules before semantic tree filtering
            rules_ = [
                tuple(rule)
                for rule in
                [Rule(r, args=args, pred=pred) for (r, pred), args in rules_]]

            # Please note: this loop changes the rule representation from
            # a list of tuples (rule, (precision, recall), pred) to
            # a list of tuples (rule, (precision, recall, pred)).
            for rule, score, pred in rules_:
                # keep only rules verifying precision_min and recall_min:
                if score[0] >= self.precision_min and score[1] >= self.recall_min:
                    if rule in self.rules_ and not self.regression:
                        # update the scores to the new mean
                        c = self.rules_[rule][2] + 1
                        b = self.rules_[rule][1] + 1. / c * (
                            score[1] - self.rules_[rule][1])
                        a = self.rules_[rule][0] + 1. / c * (
                            score[0] - self.rules_[rule][0])
                        self.rules_[rule] = (a, b, c)
                    else:
                        self.rules_[rule] = (score[0], score[1], pred)

            self.rules_ = sorted(self.rules_.items(),
                                key=lambda x: (x[1][0], x[1][1]), reverse=True)

            # Deduplicate the rule using semantic tree
            if self.max_depth_duplication is not None:
                self.rules_ = self.deduplicate(self.rules_)

            self.rules_ = sorted(self.rules_, key=lambda x: - self.f1_score(x))
            self.rules_without_feature_names_ = self.rules_

            # Replace generic feature names by real feature names
            # We change back the rule representation to
            # (rule, (precision, recall), pred).
            self.rules_ = [
                (
                    replace_feature_name(rule, self.feature_dict_),
                    perf[:-1],
                    perf[-1]
                )
                for rule, perf in self.rules_
            ]

            return self
        #def predict(self, X):