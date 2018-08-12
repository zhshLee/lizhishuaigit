import numpy
import scipy # use numpy if scipy unavailable
import scipy.linalg # use numpy if scipy unavailable

def ransac(data,model,n,k,t,d,debug=False,return_all=False):
    """fit model parameters to data using the RANSAC algorithm
{{{
Given:
    data - a set of observed data points
    model - a model that can be fitted to data points
    n - the minimum number of data values required to fit the model
    k - the maximum number of iterations allowed in the algorithm
    t - a threshold value for determining when a data point fits a model
    d - the number of close data values required to assert that a model fits well to data
Return:
    bestfit - model parameters which best fit the data (or nil if no good model is found)
iterations = 0
bestfit = nil
besterr = something really large
while iterations < k {
    maybeinliers = n randomly selected values from data
    maybemodel = model parameters fitted to maybeinliers
    alsoinliers = empty set
    for every point in data not in maybeinliers {
        if point fits maybemodel with an error smaller than t
             add point to alsoinliers
    }
    if the number of elements in alsoinliers is > d {
        % this implies that we may have found a good model
        % now test how good it is
        bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
        thiserr = a measure of how well model fits these points
        if thiserr < besterr {
            bestfit = bettermodel
            besterr = thiserr
        }
    }
    increment iterations
}
return bestfit
}}}
"""
    iterations = 0
    bestfit = None
    besterr = numpy.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n,data.shape[0])
        maybeinliers = data[maybe_idxs,:]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < t] # select indices of rows with accepted points
        alsoinliers = data[also_idxs,:]
        if len(alsoinliers) > d:
            betterdata = numpy.concatenate( (maybeinliers, alsoinliers) )
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error( betterdata, bettermodel)
            thiserr = numpy.mean( better_errs )
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = numpy.concatenate( (maybe_idxs, also_idxs) )
        iterations+=1
    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers':best_inlier_idxs}
    else:
        return bestfit

def random_partition(n,n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = numpy.arange( n_data )
    numpy.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

class LinearLeastSquaresModel:
    """linear system solved using linear least squares

    This class serves as an example that fulfills the model interface
    needed by the ransac() function.
    
    """
    def __init__(self,input_columns,output_columns,debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug
    def fit(self, data):
        A = numpy.vstack([data[:,i] for i in self.input_columns]).T
        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
        x,resids,rank,s = scipy.linalg.lstsq(A,B)
        return x
    def get_error( self, data, model):
        A = numpy.vstack([data[:,i] for i in self.input_columns]).T
        B = numpy.vstack([data[:,i] for i in self.output_columns]).T
        B_fit = scipy.dot(A,model)
        err_per_point = numpy.sum((B-B_fit)**2,axis=1) # sum squared error per row
        return err_per_point
        
def test():
    # generate perfect input data
    import xlrd
    import math
    data = xlrd.open_workbook('output_files.xls')
    table = data.sheets()[0]
    data = table.col_values(1)
    del (data[0])
    n_samples = len(data)
    n_inputs = 3
    n_outputs = 1

    # setup model
    # maxxx = max(data)
    xx = []
    xx2 = []
    xx3 = []
    data1 = []
    b = []
    for j in range(len(data)):
        xx.append(j * 50)
        # xx2.append(math.pow(j*10,2))
        xx3.append(math.pow(j * 10, 3))
        b.append(1)
    for j in data:
        data1.append(j)
    all_data = numpy.array([b,xx,xx3,data1]).T######最小二乘模型改这里，现在是 a*x^3+b*x+c = data1
    input_columns = range(n_inputs) # the first columns of the array
    output_columns = [n_inputs+i for i in range(n_outputs)] # the last columns of the array
    debug = False
    model = LinearLeastSquaresModel(input_columns,output_columns,debug=debug)


    # run RANSAC algorithm
    ransac_fit, ransac_data = ransac(all_data,model,
                                    60, 30000, 500, 0, # misc. parameters##改这里
                                     debug=debug,return_all=True)
    if 1:
        import pylab
        pylab.plot( xx[:], data1[:], 'k.', label='data' )
        xxx = (numpy.array(xx))
        data2 = (numpy.array(data1))
        pylab.plot( xxx[ransac_data['inliers']], data2[ransac_data['inliers']], 'bx', label='RANSAC data' )


        A_exact = xxx[ransac_data['inliers']]
        sort_idxs = numpy.argsort(A_exact[:])
        A_col0_sorted = A_exact[sort_idxs].reshape(A_exact.size,1) # maintain as rank-2 array

        wb = numpy.ones([A_exact.size,1])
        # A_col0_sorted2 = [A_col0_sorted[i]**2 for i in range(len(A_col0_sorted))]
        A_col0_sorted3 = [A_col0_sorted[i] ** 3 for i in range(len(A_col0_sorted))]
        mm = numpy.hstack(( wb,A_col0_sorted,A_col0_sorted3))

        pylab.plot( A_col0_sorted,
                    numpy.dot(mm,ransac_fit)[:,0],
                    label='RANSAC fit' )
        pylab.legend()
        pylab.show()


if __name__=='__main__':
    test()
