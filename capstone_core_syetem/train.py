class File_read():
    def __init__(self):
        self.user_setting_line_num = 6
    def read_user_setting(self, write_list):
        f = open("./setting/user_setting.txt", 'r')
        for i in range(self.user_setting_line_num):
            line = f.readline()
            line = int(line)
            write_list.append(line)
        f.close()
        return write_list

    def read_basic_map(self, write_list):
        f = open("./setting/basic_map.txt", 'r')
        while True:
            line = f.readline()
            if not line:
                break
            line = int(line)
            write_list.append(line)
        f.close()
        return write_list

if __name__=='__main__':
    Fr = File_read()
    user_setting_list = []
    user_setting_list = Fr.read_user_setting(user_setting_list)
    basic_map_obs_list = []
    basic_map_obs_list = Fr.read_basic_map(basic_map_obs_list)






