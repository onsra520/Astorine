from chatbot_runing import chatbot_handle

while True:
    user_id = "user_dep_trai_nhat_qua_dat"
    user_input = input("User: ")
    if user_input == "bye":
        break
    print(chatbot_handle(user_id, user_input))
    # print(reply(user_input, model_dir, data))
