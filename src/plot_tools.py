# plot_tools.py
import os
import matplotlib.pyplot as plt
import seaborn as sns

def set_paras(x_title,y_title,title=None,filename=None,file_dir='plots',has_label=False):

	'''set all the parameters in the figure and save files'''
	if has_label:
		plt.legend()
	plt.xlabel(x_title)
	plt.ylabel(y_title)
	plt.title(title)
	plt.tight_layout()
	if filename:
		full_path = os.path.join(file_dir, filename)
		plt.savefig(full_path)
		plt.close()
		# plt.show() #for testing
	else:
		plt.show()

def make_dir(file_dir):
	'''checks if the directory exists if not make one'''
	if file_dir:
		if not os.path.exists(file_dir):
			os.mkdir(file_dir)

def plot_2D(x, y, plot_count=1,title=None,x_title=None,y_title=None,label=False,filename=None,
		file_dir='plots', multi_x=False):

	'''plots inputs: x:array like of array like, y:array like of array likes,
	plot_count:int(number of plots),title:string, file_dir:string, multi_x: bool (if x values are different for each plot
	default False)'''

	make_dir(file_dir)

	if plot_count == 1:
		y = [y]

	for i in range(plot_count):
		if multi_x:
			if label:
				plt.plot(x[i],y[i],label=label[i])
			else:
				plt.plot(x[i],y[i])
		else:
			if label:
				plt.plot(x,y[i],label=label[i])
			else:
				plt.plot(x,y[i])


	set_paras(x_title, y_title, title, filename, file_dir, label)

def scatter_2D(x, y, plot_count=1,title=None,x_title=None,y_title=None,label=None,filename=None,
		file_dir='plots', multi_x=False):

	'''plots inputs: x:array like of array like, y:array like of array likes,
	plot_count:int(number of plots),title:string, file_dir:string, multi_x: bool (if x values are different for each plot
	default False)'''

	make_dir(file_dir)

	if plot_count == 1:
		x = [x]
		y = [y]

	for i in range(plot_count):
		if multi_x:
			if label:
				plt.scatter(x[i],y[i],label=label[i])
			else:
				plt.scatter(x[i],y[i])
		else:
			if label:
				plt.scatter(x,y[i],label=label[i])
			else:
				plt.scatter(x,y[i])


	set_paras(x_title, y_title, title, filename, file_dir, label)

def save_fig(fname):
	plt.savefig("plots/"+fname)
	plt.close()

def heatmap(matrix, xticklabels='', yticklabels='', x_title="Predicted",y_title="Test", annot=False, title=None, label=False,filename=None,
		file_dir='plots'):
		
	make_dir(file_dir)
	sns.heatmap(matrix, annot=annot, cmap="Blues", xticklabels=xticklabels, yticklabels=yticklabels)	
	set_paras(x_title, y_title, title, filename, file_dir, label)
