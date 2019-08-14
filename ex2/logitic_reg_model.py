class LogisticRegression():
    def __init_(n_iter=50):
        self.n_iter = n_iter

    def sigmoid(z):
        return (1/(1+ np.exp(-z)))

    def net_input(self,X):
        return np.dot(X,self.w_)

    def cost_function(self, X, y):
        z = net_input(x)
        prob = sigmoid(z)
        return (-1/len(y)) * \
               (np.dot(y.T,np.log(prob)) + \
                np.dot((1-y).T,np.log(1-prob)))
    def 
        
