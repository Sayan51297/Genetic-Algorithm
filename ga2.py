import pylab as plt
import numpy as np

def maxfunc(sample):
    x,y = sample
    return (16.*x * (1-x) * y * (1-y) * np.sin(15.*np.pi*x) * np.sin(15.*np.pi*y))**2

def rank(sample,fitness):
    fitness /= np.sum(fitness) # normalize fitness values
    sa = np.argsort(fitness) # sort the sample
    return sample[sa],np.cumsum(fitness[sa])

def select_individuals(acfitness,min_value=0.9):
    return (acfitness>min(np.random.uniform(),min_value))

def breed_crossover(parent1,parent2,parents):
    index = int(np.random.uniform(low=1,high=len(parent1)-1)) # random point between 1 and 1 is always 1
    return np.hstack([parent1[:index],parent2[index:]])

def generate_offspring(sample,good_individuals,breed_func):
    parents = sample[good_individuals] # select only good individuals
    nr_of_children = np.sum(-good_individuals) # how many individuals need to be replaced?
    Np,Ni = parents.shape
    children = np.zeros((nr_of_children,Ni)) # prepare to fill in the children
    index1 = np.array(np.random.uniform(size=nr_of_children,high=Np-1).round(),int) # random parent 1
    index2 = np.array(np.random.uniform(size=nr_of_children,high=Np-1).round(),int) # random parent 2
    for nr,(i,j) in enumerate(zip(index1,index2)):
        children[nr] = breed_func(parents[i],parents[j],parents)
    return children

def evolve(sample,eval_func,breed_func):
    fitness = eval_func(sample.T) # evaulation function needs 2xN instead of Nx2
    sample,fitness_ = rank(sample,fitness.copy())
    good = select_individuals(fitness_)
    children = generate_offspring(sample,good,breed_func)
    sample[bad] = children
    return sample

def ontype(event):

    if event.key=='enter':
        #-- retrieve the sample data from the plot: the sample is plotted in white
        for child in plt.gca().get_children():
            if hasattr(child,'get_color') and child.get_color()=='1':
                sample = np.array(child.get_data()).T # put it in the right format
                break # make sure `child` is now set to the right artist
        sample = evolve(sample,maxfunc,breed_crossover)
        child.set_data(sample[:,0],sample[:,1])

    if event.key=='i':
        sample = np.random.uniform(size=(slider.val,2),low=0,high=1) # initialize sample
        bkg = maxfunc(np.mgrid[0:1:500j,0:1:500j]) # make the background
        plt.cla()
        plt.plot(sample[:,0],sample[:,1],'o',color='1',mec='1')
        plt.imshow(bkg,extent=[0,1,0,1],aspect='auto',cmap=plt.cm.spectral)
        plt.gca().set_autoscale_on(False)

    plt.draw()
    
    if __name__ == "__main__":
    	axpop = plt.axes([0.20, 0.05, 0.65, 0.03], axisbg='lightgoldenrodyellow') # axis for the slide
    	slider = plt.Slider(axpop, 'Population', 10, 2000, valinit=500) # the slider
    	ax1 = plt.axes([0.1,0.15,0.85,0.77]) # main axis to plot the results

    	plt.gcf().canvas.mpl_connect('key_press_event',ontype) # connect the ontype function to the GUI
    	plt.show() # show the window

                            