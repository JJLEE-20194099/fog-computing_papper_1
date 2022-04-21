from config import Config
import random as rd

number_of_servers = Config.number_of_servers
server_max_frequency_by_GHZ_interval = Config.server_max_frequency_by_GHZ_interval

with open("./info.csv", "w") as output:
    output.write("Server Max Frequency (GHZ)\n")
    for _ in range(number_of_servers):
        output.write("{}\n".format(rd.randint(server_max_frequency_by_GHZ_interval[0], server_max_frequency_by_GHZ_interval[1])))

