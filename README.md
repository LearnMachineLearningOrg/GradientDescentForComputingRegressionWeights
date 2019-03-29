1. Linear regression machine learning algorithm falls under supervised learning

2. In Linear regression, we will learn a model to predict a label's value given a set of features that describe the label
Learning a model means, identifying a “hypothesis function” as described below that optimally fits the given data points and can optimally generate the label value for any new set of features
Hypothesis function: y = mx + b
So, the hypothesis function depends on the values of “m” and “b”

3. Our goal is to find the optimal values for “m” and “b” (these are also called as weights or parameters), such that the predicted value of the label will be very close to the actual value. To know whether the predicted value of the label is close the actual value or not, we use a function called “Cost Function”. Generally, Mean Square Error function is used as a cost function. The lower the value of the cost function, it indicates that the predicted value of the label is close to the actual value. So, our goal is to minimize the value of the cost function.

4. Gradient descent is an optimization algorithm that can be used to find the local minimum of a function. So, in our scenario we will use the Gradient descent algorithm to find the minimum of the MSE function. The procedure is to iterate for ‘n’ number of times, in each iteration calculating the new values for “m” and “b”, and checking whether the cost function reached the minimum

5. Math behind gradient descent:

     We will be applying partial derivatives with respect to both m and b to the cost function to point us to the lowest point. A derivative of zero means you are at either a local minima or maxima. Which means that the closer we get to zero, the better. When we reach close to, if not, zero with our derivatives, we also inevitably get the lowest value for our cost function.
     	     
     We use a hyper-parameter called learning rate that defines how fast gradient descent finds the optimal parameters (‘m’ and ‘b’)
          A larger value for the learning rate than which is required means that the steps we take are too big and that you might miss the lowest point entirely
          A smaller value for the learning rate than which is required means that the steps we take are too small and it might take long time to converge
