with open('target_path.txt', mode='r', encoding='utf-8') as f:
    target_path = f.readline().replace('\n', '')
get_hist_num = target_path.replace(' ', '').split("hist")[0].split("/")[-1]
print("!!! {}hist !!!".format(get_hist_num)) #test

with open(".memory_size", "w", encoding="utf-8") as f:
    inp = get_hist_num
    
    if inp.isnumeric():
        memory_size = inp
        print("memory_size output")
        f.write(memory_size)
    else:
        print("Not Numeric")
        f.write("")
