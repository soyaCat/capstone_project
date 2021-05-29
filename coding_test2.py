class DFS():
    def __init__(self):
        self.dic = {}
        self.visited = []

    def make_connect_map(self, num_com, connect_num):
        for i in range(num_com):
            self.dic[i + 1] = set()
        for i in range(connect_num):
            connect_inf = input()
            a, b = map(int, connect_inf.split(' '))
            self.dic[a].add(b)
            self.dic[b].add(a)
        return self.dic

    def dfs_algorithm(self, start, dic):
        for i in dic[start]:
            if i not in self.visited:
                self.visited.append(i)
                self.dfs_algorithm(i, dic)

if __name__=="__main__":
    num_com=int(input())
    connect_num = int(input())
    dfs_function = DFS()
    dic = dfs_function.make_connect_map(num_com, connect_num)

    dfs_function.dfs_algorithm(1, dic)
    print(len(dfs_function.visited)-1)