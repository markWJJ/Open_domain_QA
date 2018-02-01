import  json

json_data=json.load(open("./train-v1.1.json",'r'))

max_question_len=30
max_para_len=100

write_file=open("SQUQA_train1.txt",'w')

for index,ele in enumerate(json_data["data"]):
    if index ==1:
        for e in ele["paragraphs"]:
            #print(e)
            for qas in e["qas"]:
                if len(str(e["context"]).replace("\n","").split(" "))>max_para_len:
                    pass
                else:
                    if len(str(qas["question"]).replace("\n","").split(" "))>max_question_len:
                        pass
                    else:


                        #start_id=int(qas["answers"][0]["answer_start"])
                        print("correct",qas["answers"][0]["text"])
                        #print(len(str(e["context"])))
                        #print(qas["answers"][0])
                        #print(str(e["context"])[start_id:start_id+len(qas["answers"][0]["text"])])
                        #print(e["context"])
                        answer_len=len(str(qas["answers"][0]["text"]).split(" "))
                        context_list=str(e["context"]).replace("\n","").split(" ")
                        start_id=""
                        end_id=""
                        for i in range(0,len(context_list)-answer_len):
                            if (" ".join(context_list[i:i+answer_len])).replace(".","")==qas["answers"][0]["text"] and len(" ".join(context_list[0:i]))+1==qas["answers"][0]["answer_start"]:
                                print(qas["answers"][0]["text"])
                                start_id=str(i)
                                end_id=str(i+answer_len)
                        if start_id!="" and end_id!="":
                            write_file.write(str(qas["question"]).replace("\n", ""))
                            write_file.write("\t\t")
                            write_file.write(str(e["context"]).replace("\n", ""))
                            write_file.write("\t\t")
                            write_file.write(str(start_id)+"-"+str(end_id))
                            write_file.write("\n")
                # print(e["context"])
                # print(qas["answers"])
                # print(qas["question"])
                # print("\n")
        # for e in ele["paragraphs"]:
        #     print(e["context"])
        #     print("\n")
        #     print(e)
        #     # print(e['context'])
        #     # print(e["question"])
        #     # print(e["answers"])
