def create_list(items_file, all_item_label_filenname, test_item_label_filename, tr_file, te_file):
	all_item_label = {}
	item_idx = {}
	test_item_label = {}
	train_item_label = {}
	all_items = [l.strip() for l in open(items_file, 'r').readlines()]
	for idx, item  in enumerate(all_items):
		item_idx[item] = idx

	with open(all_item_label_filenname, 'r') as f:
		for line in f:
			line = line.strip()
			item = line.split(' ')[0]
			label = line.split(' ')[1]
			all_item_label[item] = label

	with open(test_item_label_filename, 'r') as f:
		for line in f:
			line = line.strip()
			item = line.split(' ')[0]
			label = line.split(' ')[1]
			test_item_label[item] = label

	train_items = []	
	test_items = []	
	for item in all_items:
		if item in test_item_label.keys():
			test_items.append(item)
		else:
			train_items.append(item)
	
	
	print(len(train_items))
	print(len(test_items))
	with open(tr_file, 'w') as f:
		for item in train_items:
			line = item + ' ' + all_item_label[item] + '\n'
			f.write(line)

	with open(te_file, 'w') as f:
		for item in test_items:
			line = item + ' ' + test_item_label[item] + '\n'
			f.write(line)


create_list('items.txt', 'all_items_label.txt', 'test_items_label.txt', 'trainlist01_str.txt', 'testlist01_str.txt')
