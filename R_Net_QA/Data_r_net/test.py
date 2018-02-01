
write_file=open("train_new.txt",'w')

with open("./train1.txt",'r') as read_file:

    for ele in read_file:
        ele=ele.replace("\n","")
        que=ele.split("\t\t")[0]
        passage=ele.split("\t\t")[1]
        passage="no_answer/0 "+passage
        sent=[]
        for ele in passage.split(" "):
            if ele:
                sent.append(ele.split("/")[0])
        labe1=""
        passes=passage.split(" ")
        for i in range(len(passes)):
            if passes[i]:
                pas=passes[i].split("/")
                begin_index=""
                end_index=""
                #print(passes[i])
                if pas[1]=="0":
                    pass
                else:
                    if pas[1]=="S":
                        labe1=str(i)+"-"+str(i)
                        break
                    elif pas[1]=="B":
                        begin_index=str(i)
                        for j in range(i,len(passes)):
                            if passes[j].split("/")[1] =="E":
                                end_index=str(j)
                                break
                        labe1=begin_index+"-"+end_index
                        break
        if labe1 == "":
            labe1="0-0"
        write_file.write(que)
        write_file.write("\t\t")
        write_file.write(" ".join(sent))
        write_file.write("\t\t")
        write_file.write(labe1)
        write_file.write("\n")




