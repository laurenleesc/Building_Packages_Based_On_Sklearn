from sklearn.linear_model import LogisticRegression

class Hdlr(LogisticRegression):
    '''This is our beta correction code'''

    def fix_betas(self):
        coef = self.coef_.copy()
        mask = self.coef_ >= 0
        coef[mask]  -= 1.0
        coef[~mask]  += 1.0
        self.coef_ = coef
        

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    clf = Hdlr()
    clf.fit(X_train, y_train)
    clf.fix_betas()
    out = clf.predict(X_test)

    correct = (sum([a==b for (a,b) in zip(out, y_test)]))
    print(correct/len(out), correct, len(out))
    print(clf.get_params())
