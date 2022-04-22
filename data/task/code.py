from generate import Generate
import os

number_of_online_user = 10
generate_tool = Generate(number_of_online_user)


def run(number_of_online_user):
    generate_tool = Generate(number_of_online_user)

    if os.path.exists("./timeslot_datasets_{}".format(number_of_online_user)) == False:
        os.makedirs("./timeslot_datasets_{}".format(number_of_online_user))

    generate_tool.generate_task_info()

# run(10)
run(1000)

