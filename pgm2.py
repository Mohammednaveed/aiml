import os
import time

def get_node (mark_road,extended):
    temp=[0]
    i=0
    while 1:
        current=temp[i]
        if current not in extended:
            return current
        else:
            for child in mark_road[current]:
                if child not in temp:
                    temp.append(child)     
            i+=1

def get_current(s,nodes_tree):
    if len(s)==1:
        return s[0]
    for node in s:
        flag=True
        for edge in nodes_tree(node):
            for child_nod in edge:
                if child_nod in s:
                    flag=False
        if flag:
            return node
        
def get_pre(current,pre,pre_list):
    if current==0:
        return
    for pre_node in pre[current]:
        if pre_node not in pre_list:
            pre_list.append(pre_node)
    return

def ans_print(mark_rode,node_tree):
    print("The final connection is as follows")
    temp=[0]
    while temp:
        time.sleep(1)
        print(f"[{temp[0]}]-->{mark_rode[temp[0]]}")
        for child in mark_rode[temp[0]]:
            if node_tree[child]!=[[child]]:
                temp.append(child)
        temp.pop(0)
    time.sleep(5)
    os.system('cls')
    return

def AOstar(nodes_tree,h_val):
    futility=0xfff
    extended=[]
    choice=[]
    mark_rode={0:None}
    solved={}
    pre={0:[]}
    for i in range(1,9):
        pre[i]=[]
    for i in range(len(nodes_tree)):
        solved[i]=False
    os.system('cls')
    print("The connection process is as follows")
    time.sleep(1)
    while not solved[0] and h_val[0]<futility:
        node=get_node(mark_rode,extended)
        extended.append(node)
        if nodes_tree[node] is None:
            h_val[node]=futility
            continue
        for suc_edge in nodes_tree[node]:
            for suc_node in suc_edge:
                if nodes_tree[suc_node]==[[suc_node]]:
                    solved[suc_node]=True
        s=[node]
        while s:
            current=get_current(s, nodes_tree)
            s.remove(current)
            origen_h=h_val[current]
            origen_s=solved[current]
            min_h=0xfff
            for edge in nodes_tree[current]:
                edge_h=0
                for node in edge:
                    edge_h+=h_val[node]+1
                if edge_h<min_h:
                    min_h=edge_h
                    h_val[current]=min_h
                    mark_rode[current]=edge
            if mark_rode[current] not in choice:
                choice.append(mark_rode[current])
                print(f"[{current}]--{mark_rode[current]}")
                time.sleep(1)
            for child_node in mark_rode[current]:
                pre[child_node].append(current)
            solved[current]=True
            for node in mark_rode[current]:
                solved[current]=solved[current] and solved[node]
            if origen_s!=solved[current] or origen_h!=h_val[current]:
                pre_list=[]
                if current!=0:
                    get_pre(current,pre,pre_list)
                s.extend(pre_list)
    if not solved[0]:
        print("The query failed, the path could not be found")
    else:
        ans_print(mark_rode, nodes_tree)
        return
if __name__=="__main__":
    nodes_tree={}
    nodes_tree[0]=[[1],[4,5]]
    nodes_tree[1]=[[2],[3]]
    nodes_tree[2]=[[3],[2,5]]
    nodes_tree[3]=[[5,6]]
    nodes_tree[4]=[[5],[8]]
    nodes_tree[5]=[[6],[7,8]]
    nodes_tree[6]=[[7,8]]
    nodes_tree[7]=[[7]]
    nodes_tree[8]=[[8]]
    h_val=[3,2,4,4,1,1,2,0,0]
    AOstar(nodes_tree, h_val)