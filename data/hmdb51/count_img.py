import glob
def count_img_rgb(pre_list, rgb_list):
	of = open(rgb_list, 'w')
	with open(pre_list, 'r') as f:
		for line in f:
			item = line.strip().split(' ')[0]
			cls = line.strip().split(' ')[2]
			img_dir = line.strip().split(' ')[1]
			num = len(glob.glob(img_dir+'/*.jpg'))
			new_line = ' '.join([item, img_dir, str(num), cls])+'\n'
			print(new_line)
			of.write(new_line)


def count_img_flow(pre_list, flow_list):
	of = open(flow_list, 'w')
	with open(pre_list, 'r') as f:
		for line in f:
			item = line.strip().split(' ')[0]
			cls = line.strip().split(' ')[2]
			img_dir = line.strip().split(' ')[1].replace('{:s}', 'u')
			print(img_dir)# = line.strip().split(' ')[1].replace('{:s}', 'u')
			num = len(glob.glob(img_dir+'/*.jpg'))
			new_line = ' '.join([item, img_dir.replace('u', '{:s}'), str(num), cls])+'\n'
			#print(new_line)
			of.write(new_line)
			



if __name__ == '__main__':
#	count_img_rgb('pre_rgb.txt', 'right_rgb.txt')
	count_img_flow('pre_flow.txt', 'right_flow.txt')
