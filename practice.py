from multiprocessing import Pool

def foo(num):
	return (num, num*num)

if __name__=="__main__":
	pool = Pool()
	results = []
	for i in range(10):
		results.append(pool.apply_async(foo, args=[i]))
	i1_list = []
	i2_list = []
	for result in results:
		i1, i2 = result.get()
		i1_list.append(i1)
		i2_list.append(i2)
	print i1_list
	print i2_list