import os, sys
import json5
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.append(str(root))
from nlp.helper.qbuilder import assembler
from nlp.helper.ncomp import rlst, srlst, clst, glst, rrlst, dtlst, sslst, blst

paths = {
    "intents": os.path.abspath(f"{root}/intents/intents.json"),
    "qfragments": os.path.abspath(f"{root}/intents/qfragments.json"),
}

intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "Hello",
                "Hey",
                "Hi",
                "Hi there",
                "Good morning",
                "Good afternoon",
                "Good evening",
                "Is anyone there?",
                "Anyone there?",
                "Hey bot",
                "Hi bot",
                "Hello bot",
                "Yo",
                "What's up?",
                "Greetings",
                "Howdy",
            ],
            "responses": [
                "Hello!",
                "Good to see you again!",
                "Hi there, What can I do for you?",
                "Hello, how can I help?",
                "Hi there! How can I assist you today?",
                "Hello! How can I help you find the perfect laptop?",
                "Hey! Ready to explore some great laptops?",
            ],
        },
        {
            "tag": "goodbye",
            "patterns": [
                "Bye",
                "See you later",
                "Goodbye",
                "see ya",
                "I am leaving",
                "Have a Good day",
                "Have a nice day",
                "Bye! Come back again soon.",
                "See you next time",
                "Take care, goodbye",
                "I'm signing off",
                "Until next time",
                "Catch you later",
                "Talk to you soon",
                "Bye for now",
                "I gotta go, bye!",
                "Time to say bye, take care!" "Bye for now",
                "I gotta go, bye!",
                "Time to say bye, take care!",
                "Bye bye, see you soon!",
                "I'm off now, bye!",
                "Catch you later, bye!" "I'm logging off now, bye!",
                "Alright, I'm leaving, bye!",
                "Bye now, see you next time!",
                "Bye, and take care of yourself!",
                "It was great talking, bye!",
            ],
            "responses": [
                "Goodbye! Have a great day!",
                "See you soon!",
                "Until next time!",
                "Bye! Come back again soon.",
            ],
        },
        {
            "tag": "Astorine",
            "patterns": [
                "What can you tell me about Astorine?",
                "I'm curious about Astorine.",
                "Could you elaborate on what Astorine is?",
                "What is the story behind Astorine?",
                "Can you provide further details on Astorine?",
                "I'd like to learn more about Astorine.",
                "Could you introduce me to Astorine?",
                "Please give me a rundown on Astorine." "What is Astorine?",
                "Tell me about Astorine",
                "Explain Astorine",
                "Who is Astorine?",
                "Astorine?",
                "Define Astorine",
                "Please describe Astorine",
                "Can you tell me more about Astorine?",
                "I want to know about Astorine",
                "What does Astorine do?",
                "How would you explain Astorine?",
                "Could you provide some details about Astorine?",
                "Introduce yourself, Astorine",
                "Give me information on Astorine",
            ],
            "responses": [
                "Astorine is a Chatbot that helps you to find the best Laptop for you. It is developed by Tran Tien",
                "Astorine is an AI-driven chatbot designed by Tran Tien to assist you in finding the ideal laptop.",
                "Astorine is a chatbot created by Tran Tien to help you find the perfect laptop.",
                "I am Astorine, your assistant. I am here to help you find the perfect laptop.",
                "I am Astorine, a chatbot designed to help you find the perfect laptop.",
            ],
        },
        {
            "tag": "developer",
            "patterns": [
                "Who is Tran Tien?",
                "Who is the developer of Astorine?",
                "Who is the founder of Astorine?",
                "Who is the creator of Astorine?",
                "Who is the owner of Astorine?",
                "Tran Tien",
                "developer",
                "founder",
                "creator",
                "owner",
                "Tell me about Tran Tien.",
                "Can you tell me who Tran Tien is?",
                "I want to know more about the developer of Astorine.",
                "Please explain who the developer of Astorine is.",
                "Give me details about the founder of Astorine.",
                "Do you know who created Astorine?",
                "Astorine was developed by whom?",
                "Who owns Astorine?",
                "What do we know about the creator of Astorine?",
                "Identify the developer of Astorine.",
                "I need info on Tran Tien, the founder of Astorine.",
                "Please tell me about the owner of Astorine.",
                "Can you provide background information on Tran Tien?",
                "Tell me more about the Astorine developer.",
            ],
            "responses": [
                "Tran Tien is a student at FPT University. He is my master",
                "Tran Tien is a student at FPT University. He is the creator of Astorine.",
                "Tran Tien is a student at FPT University. He is the owner of Astorine.",
            ],
        },
        {
            "tag": "name",
            "patterns": [
                "How shall I address you?",
                "What do people usually call you?",
                "Please tell me the name you go by.",
                "What's the name you respond to?",
                "May I know how you are addressed?" "What is your name?",
                "What should I call you?",
                "Do you have a name?",
                "Name",
                "Could you please tell me your name?",
                "May I know your name?",
                "Who am I talking to?",
                "Introduce yourself",
                "What's your handle?",
                "How do you go by?",
                "Tell me your name",
                "Can you share your name?",
            ],
            "responses": [
                "You can call me Astorine.",
                "I am Astorine!",
                "I am Astorine, your assistant.",
            ],
        },
        {
            "tag": "help",
            "patterns": [
                "Can you help me?",
                "What can you do for me?",
                "Can you help me?",
                "What do you offer?",
                "What are your options?",
                "Help",
                "I need help",
                "Help me",
                "Could you assist me?",
                "I require assistance",
                "Please help",
                "I need some support",
                "Is there any help available?",
                "Do you provide help?",
                "What services do you offer?",
                "How can you assist me?",
                "Assist me please",
                "I am in need of help",
                "Could you give me some help?",
                "Please provide assistance",
                "Assistance!",
                "I'm stuck, help!",
                "I need some help solving a problem.",
                "I'm really struggling with this; can you help me out?",
                "I'm facing a significant issue right now and need immediate assistance, could you please help me?",
            ],
            "responses": [
                "🚀 Let's kick off your search for the perfect laptop! Which brand are you leaning towards?",
                "🔎 Okay, I am here to help you! Do you have a favorite laptop brand in mind?",
                "💡 Ready to explore? Tell me the brand you are interested in!",
            ],
        },
        {
            "tag": "thanks",
            "patterns": [
                "Thanks",
                "Thank you",
                "That is helpful",
                "Thank is a lot!",
                "Thanks a lot",
                "Many thanks",
                "I really appreciate it",
                "I appreciate your help",
                "Thanks so much",
                "Thank you very much",
                "I am grateful",
                "I appreciate this",
            ],
            "responses": ["Happy to help!", "Any time!", "My pleasure"],
        },
        {
            "tag": "items",
            "patterns": [
                "Which items do you have?",
                "What kinds of items are there?",
                "What do you sell?",
                "Can you list the products you offer?",
                "Could you show me the items available?",
                "Do you have any products for sale?",
                "What items are on offer?",
                "Tell me about the items you stock.",
                "What's available for purchase?",
                "What products do you have in stock?",
                "List your items please.",
                "What types of items can I buy?",
                "Could you provide a list of items?",
                "What merchandise do you sell?",
                "Do you offer any goods for sale?",
                "Please show me the inventory.",
            ],
            "responses": [
                "We offer a curated collection of high-end laptops, featuring both luxurious and stylish designs.",
                "We specialize in the sale of premium laptops. Our inventory boasts a diverse selection, encompassing luxurious and stylish models.",
                "Experience the pinnacle of laptop technology. Explore our diverse collection of high-end laptops, featuring luxurious designs and unparalleled style.",
            ],
        },
        {
            "tag": "payments",
            "patterns": [
                "Do you take credit cards?",
                "Do you accept Mastercard?",
                "Can I pay with Paypal?",
                "Are you cash only?",
                "Do you accept Visa?",
                "Can I pay with American Express?",
                "Do you take debit cards?",
                "Is Apple Pay accepted?",
                "Do you support Google Pay?",
                "What payment methods do you accept?",
                "Are there alternative payment options available?",
                "Do you allow bank transfers?",
                "Can I pay using a mobile wallet?",
                "Do you accept cryptocurrency?",
                "Is contactless payment available?",
                "Can I pay with a gift card?",
                "Do you offer installment plans?",
                "What are the accepted payment types?",
                "How can I pay for my order?",
            ],
            "responses": [
                "We accept all payment methods, even Bitcoin payments",
                "We offer a variety of payment options, including Bitcoin.",
                "Pay your way. We accept Bitcoin and all major credit cards.",
            ],
        },
        {
            "tag": "delivery",
            "patterns": [
                "How long does delivery take?",
                "How long does shipping take?",
                "When do I get my delivery?",
                "What is the estimated delivery time?",
                "How many days until my package arrives?",
                "When will my order be delivered?",
                "Can you provide the shipping duration?",
                "What are the delivery times?",
                "How quickly is the shipping?",
                "Do you offer expedited shipping?",
                "How fast will my order arrive?",
                "What's the delivery schedule?",
                "When can I expect my shipment?",
                "Please tell me the delivery timeframe.",
                "What are the shipping times for my order?",
            ],
            "responses": [
                "We guarantee your order will be delivered within 2-5 hours.",
                "You can count on receiving your order within the next 2-5 hours.",
                "Your order is just a few hours away from you.",
            ],
        },
        {
            "tag": "funny",
            "patterns": [
                "Tell me a joke!",
                "Tell me something funny!",
                "Do you know a joke?",
                "Joke",
            ],
            "responses": [
                "You want to be a masterchef? buy the 'Laptop MSI Gaming GF63 Thin' furnace, you can cook when using it :D",
                "I have been a fan of the Asus ROG Zephyrus for a solid 20 years. After learning that the ROG Zephyrus is the most powerful gaming laptop of all time, I feel like a king wearing royal robes whenever I take it out for breakfast in the morning. Walking down the street, even dogs look scared and do not dare to bark. In the afternoon, I dominate a few matches, and by evening, I am glowing with victory. At night, when I hit the town, girls compete to take photos with me. My exes all suddenly want to rekindle old flames. My friends look at me with respect and bow their heads in admiration. Back home, my family is proud, and the neighbors are green with envy because they know I have been a loyal Asus ROG Zephyrus fan for 20 years.",
            ],
        },
        {
            "tag": "gpu_question",
            "patterns": [
                "What is a GPU?",
                "Can you explain what a GPU is?",
                "What does GPU stand for?",
                "Define GPU",
                "Tell me about GPU",
                "What are GPUs used for?",
                "How does a GPU work?",
                "What's the role of a GPU in a computer?",
                "What is the function of a GPU?",
                "Give me an explanation of GPU",
                "Explain GPU",
            ],
            "responses": [
                "A GPU, or Graphics Processing Unit, is a specialized processor designed to accelerate the creation and rendering of images, animations, and video for output to a display. It performs parallel processing to handle complex computations efficiently.",
                "GPU stands for Graphics Processing Unit. It is a critical component in modern computing that offloads graphics-related tasks from the CPU, enabling smoother rendering of visuals in games, video editing, and professional applications.",
                "A GPU is designed to process large amounts of visual data quickly. By performing many calculations simultaneously, it helps deliver high-quality graphics and supports tasks such as machine learning, scientific simulations, and video processing.",
            ],
        },
        {
            "tag": "cpu_question",
            "patterns": [
                "What is a CPU?",
                "Can you explain what a CPU is?",
                "Define CPU",
                "What does CPU stand for?",
                "Tell me about CPU",
                "How does a CPU work?",
                "What's the role of a CPU in a computer?",
                "What is the function of a CPU?",
                "Give me an explanation of CPU",
                "Explain CPU",
            ],
            "responses": [
                "A CPU, or Central Processing Unit, is the primary component of a computer that performs most of the processing inside a computer. It executes instructions from programs and handles tasks such as arithmetic, logic, and input/output operations.",
                "CPU stands for Central Processing Unit. It is often considered the brain of the computer because it carries out the commands of a computer program by performing basic arithmetic, logical, control, and input/output operations.",
                "The CPU is the core processor of a computer system. It processes instructions from software and performs the essential calculations and data manipulations necessary for the computer to function effectively.",
            ],
        },
        {
            "tag": "ram_question",
            "patterns": [
                "What is RAM?",
                "Can you explain what RAM is?",
                "Define RAM",
                "What does RAM stand for?",
                "Tell me about RAM",
                "How does RAM work?",
                "What's the role of RAM in a computer?",
                "What is the function of RAM?",
                "Give me an explanation of RAM",
                "Explain RAM",
            ],
            "responses": [
                "RAM stands for Random Access Memory. It is a type of volatile memory used to store data temporarily while your computer is running, allowing quick access to active programs and processes.",
                "Random Access Memory (RAM) is the short-term memory of a computer where data is stored for immediate use. It enables the system to run multiple applications smoothly by quickly accessing necessary information.",
                "RAM is crucial for system performance as it holds the data and instructions that a computer processor needs in real time. Its volatile nature means that all stored data is lost when the power is off.",
            ],
        },
    ]
}

phrases = [
    "I want a laptop that has [component]",
    "I need a laptop with [component]",
    "I'm looking for a laptop that includes [component]",
    "I'm seeking a laptop featuring [component]",
    "I require a laptop containing [component]",
    "I desire a laptop equipped with [component]",
    "I'm searching for a laptop that offers [component]",
    "Could you recommend a laptop with [component]?",
    "Do you know any laptops that have [component]?",
    "Can you suggest a laptop featuring [component]?",
    "Are there any laptops available with [component]?",
    "Which laptops come with [component]?",
    "What laptops would you recommend that have [component]?",
    "Where can I find laptops equipped with [component]?",
    "How can I find a laptop that includes [component]?",
    "Any ideas for laptops containing [component]?",
    "What are my options for laptops with [component]?",
    "I'm considering a laptop that has [component]",
    "I'm contemplating buying a laptop with [component]",
    "I'm thinking about getting a laptop featuring [component]",
    "I'm surveying the market for laptops providing [component]",
    "I'm window-shopping for laptops that have [component]",
    "My preference is for a laptop with [component]",
    "I prefer laptops that feature [component]",
    "I favor laptops containing [component]",
    "I'm partial to laptops equipped with [component]",
    "I'm inclined toward laptops that include [component]",
    "I'm drawn to laptops offering [component]",
    "I'm attracted to laptops with [component]",
    "I'm keen on laptops that have [component]",
    "I'm fond of laptops providing [component]",
    "I gravitate toward laptops containing [component]",
    "A laptop with [component] is what I need",
    "What I require is a laptop featuring [component]",
    "A laptop containing [component] is essential for me",
    "For my purposes, I need a laptop equipped with [component]",
    "My work demands a laptop that includes [component]",
    "My requirements include a laptop offering [component]",
    "My situation calls for a laptop with [component]",
    "I can't do without a laptop that has [component]",
    "It's imperative that my laptop provides [component]",
    "A laptop containing [component] is a must for me",
    "My goal is to find a laptop with [component]",
    "I aim to purchase a laptop that has [component]",
    "I intend to buy a laptop featuring [component]",
    "I plan to acquire a laptop containing [component]",
    "I'm determined to get a laptop equipped with [component]",
    "I hope to secure a laptop that includes [component]",
    "I aspire to own a laptop offering [component]",
    "I'm committed to finding a laptop with [component]",
    "I'm set on getting a laptop that provides [component]",
    "I'm resolved to purchase a laptop having [component]",
    "Currently, I'm seeking a laptop with [component]",
    "At the moment, I need a laptop that has [component]",
    "Right now, I'm after a laptop featuring [component]",
    "Presently, I'm in need of a laptop containing [component]",
    "These days, I'm looking for a laptop equipped with [component]",
    "Lately, I've been searching for a laptop that includes [component]",
    "Recently, I've developed an interest in laptops offering [component]",
    "This week, I'm focusing on laptops with [component]",
    "Today, my priority is finding a laptop that provides [component]",
    "Soon, I'll be purchasing a laptop having [component]",
    "Someone looking for performance would want a laptop with [component]",
    "A person in my position would need a laptop that has [component]",
    "Anyone in my field would benefit from a laptop featuring [component]",
    "Most professionals would choose a laptop containing [component]",
    "A typical user might prefer a laptop equipped with [component]",
    "The average consumer might look for a laptop that includes [component]",
    "Many shoppers are seeking laptops offering [component]",
    "Customers often inquire about laptops with [component]",
    "Tech enthusiasts typically demand laptops that provide [component]",
    "Students generally require laptops having [component]"
    "Someone looking for performance would want a laptop with [component]",
    "A person in my position would need a laptop that has [component]",
    "Anyone in my field would benefit from a laptop featuring [component]",
    "Most professionals would choose a laptop containing [component]",
    "A typical user might prefer a laptop equipped with [component]",
    "The average consumer might look for a laptop that includes [component]",
    "Many shoppers are seeking laptops offering [component]",
    "Customers often inquire about laptops with [component]",
    "Tech enthusiasts typically demand laptops that provide [component]",
    "Students generally require laptops having [component]",
    "People who work remotely would appreciate a laptop with [component]",
    "Gamers would definitely want a laptop that has [component]",
    "Content creators would search for a laptop featuring [component]",
    "Programmers would insist on laptops containing [component]",
    "Business professionals would invest in laptops equipped with [component]",
    "Graphic designers would prioritize laptops that include [component]",
    "Video editors couldn't work without laptops offering [component]",
    "Data scientists would rely heavily on laptops with [component]",
    "Engineers would specifically request laptops that provide [component]",
    "Architects would look for laptops boasting [component]",
    "Musicians would gravitate toward laptops having [component]",
    "Digital nomads would depend on laptops containing [component]",
    "Photographers would benefit from laptops sporting [component]",
    "Teachers would find value in laptops that possess [component]",
    "Healthcare professionals would utilize laptops featuring [component]",
    "Financial analysts would require laptops that incorporate [component]",
    "IT professionals would naturally select laptops with [component]",
    "Researchers would seek out laptops equipped with [component]",
    "Writers would appreciate laptops that offer [component]",
    "College students would shop for laptops including [component]",
    "Someone concerned about future-proofing would invest in a laptop with [component]",
    "A budget-conscious buyer might still prioritize a laptop that has [component]",
    "Anyone working with large files would need a laptop featuring [component]",
    "Most frequent travelers would opt for laptops containing [component]",
    "A power user couldn't settle for less than a laptop equipped with [component]",
    "The discerning consumer would investigate laptops that include [component]",
    "Many professionals in creative fields depend on laptops offering [component]",
    "Customers with technical backgrounds often choose laptops with [component]",
    "Tech reviewers consistently recommend laptops that provide [component]",
    "Students in STEM fields typically need laptops having [component]",
    "People who multitask heavily would benefit from a laptop with [component]",
    "Gamers on a budget might still insist on a laptop that has [component]",
    "Anyone working with AI applications would require a laptop featuring [component]",
    "Most small business owners would invest in laptops containing [component]",
    "A typical developer would seek out a laptop equipped with [component]",
    "The average 3D artist would only consider laptops that include [component]",
    "Many entry-level professionals are recommended laptops offering [component]",
    "Customers planning for longevity often select laptops with [component]",
    "Tech-savvy parents typically choose laptops that provide [component]",
    "Students studying computer science inevitably need laptops having [component]",
    "People in the finance sector would calculate the value of a laptop with [component]",
    "Gamers streaming their gameplay would require a laptop that has [component]",
    "Anyone attending virtual meetings regularly would value a laptop featuring [component]",
    "Most consultants on the go would invest in laptops containing [component]",
    "A typical analyst would process data faster with a laptop equipped with [component]",
    "The average video content creator wouldn't compromise on laptops that include [component]",
    "Many IT departments standardize on laptops offering [component]",
    "Customers with accessibility needs often prefer laptops with [component]",
    "Tech industry professionals consistently recommend laptops that provide [component]",
    "Students doing machine learning projects definitely require laptops having [component]"
    "Compared to other options, I prefer a laptop with [component]",
    "Unlike my previous device, I need a laptop that has [component]",
    "In contrast to basic models, I'm seeking a laptop featuring [component]",
    "Relative to standard configurations, I want a laptop containing [component]",
    "Against the competition, I'd choose a laptop equipped with [component]",
    "If reliability is important, then a laptop that includes [component] is essential",
    "When considering long-term use, a laptop offering [component] makes sense",
    "Provided the price is reasonable, I want a laptop with [component]",
    "As long as it's within budget, I need a laptop that provides [component]",
    "Unless there are better alternatives, I'm looking for a laptop having [component]",
    "Due to my work requirements, I need a laptop with [component]",
    "Because of my frequent travel, I'm seeking a laptop that has [component]",
    "Owing to the nature of my projects, I require a laptop featuring [component]",
    "On account of my specialized needs, I want a laptop containing [component]",
    "Given my professional demands, I'm after a laptop equipped with [component]",
    "According to tech experts, the best laptops include [component]",
    "Based on professional reviews, I should get a laptop that offers [component]",
    "Tech specialists recommend laptops with [component]",
    "Industry standards suggest laptops that provide [component]",
    "Following expert advice, I'm considering a laptop having [component]",
    "Above all else, I absolutely must have a laptop with [component]",
    "The single most important feature I need is a laptop that has [component]",
    "Without a doubt, I'm committed to finding a laptop featuring [component]",
    "By all means, I insist on a laptop containing [component]",
    "Under no circumstances will I compromise on a laptop equipped with [component]",
    "Looking ahead, I'll need a laptop that includes [component]",
    "In the long run, I'd benefit from a laptop offering [component]",
    "With future needs in mind, I'm searching for a laptop with [component]",
    "Planning for tomorrow, I want a laptop that provides [component]",
    "Anticipating upcoming projects, I require a laptop having [component]",
    "Throughout my career, I've always relied on laptops with [component]",
    "After my last laptop failed, I realized I need one that has [component]",
    "My experience has taught me to look for laptops featuring [component]",
    "Having used various devices, I've come to prefer laptops containing [component]",
    "Over the years, I've learned the value of laptops equipped with [component]",
    "The ultimate laptop for my needs would include [component]",
    "The ideal computing solution would be a laptop that offers [component]",
    "The perfect match for my requirements is a laptop with [component]",
    "The most suitable option would be a laptop that provides [component]",
    "The best possible choice for me is a laptop having [component]",
    "Would a laptop having [component] be within my budget?",
    "I'm curious about laptops with [component]",
    "Online forums recommend laptops containing [component]",
    "My colleague swears by laptops equipped with [component]",
    "Industry publications favor laptops that include [component]",
    "Benchmark tests highlight laptops offering [component]",
    "Consumer reports praise laptops with [component]",
    "IT departments typically deploy laptops that provide [component]",
    "My university recommends laptops having [component]",
    "A laptop with [component] would streamline my workflow",
    "Using a laptop that has [component] would boost my productivity",
    "I'd complete projects faster with a laptop featuring [component]",
    "My efficiency would improve with a laptop containing [component]",
    "I could better meet deadlines with a laptop equipped with [component]",
    "Creative tasks would be easier on a laptop that includes [component]",
    "I'd experience fewer limitations with a laptop offering [component]",
    "My computing experience would improve with a laptop with [component]",
    "I'd enjoy using a laptop that provides [component]",
    "Work would be more manageable with a laptop having [component]",
    "Specifically, I'm interested in a laptop with [component]",
    "To be precise, I need a laptop that has [component]",
    "In particular, I'm after a laptop featuring [component]",
    "Above all, I want a laptop containing [component]",
    "Primarily, I require a laptop equipped with [component]",
    "First and foremost, I'm seeking a laptop that includes [component]",
    "My main concern is finding a laptop offering [component]",
    "My chief requirement is a laptop with [component]",
    "The critical feature I need is a laptop that provides [component]",
    "The decisive factor for me is a laptop having [component]",
    "After careful consideration, I've decided on a laptop with [component]",
    "Having weighed all options, I prefer a laptop that has [component]",
    "Upon reflection, I'm convinced I need a laptop featuring [component]",
    "After thorough research, I'm focused on laptops containing [component]",
    "Based on my analysis, I've selected laptops equipped with [component]",
    "Following extensive comparison, I'm targeting laptops that include [component]",
    "After consulting with experts, I'm pursuing laptops offering [component]",
    "Having tested several models, I'm set on laptops with [component]",
    "With all factors considered, I'm choosing laptops that provide [component]",
    "After evaluating the market, I'm settling on laptops having [component]",
    "For optimal results, I should use a laptop with [component]",
    "To maximize efficiency, I'd need a laptop that has [component]",
    "For best performance, the ideal is a laptop featuring [component]",
    "To excel in my field, I require a laptop containing [component]",
    "For seamless operation, I'm seeking a laptop equipped with [component]",
    "To handle intensive tasks, I want a laptop that includes [component]",
    "For professional purposes, I need a laptop offering [component]",
    "To maintain competitive edge, I'm looking at laptops with [component]",
    "For reliable computing, I depend on laptops that provide [component]",
    "To ensure compatibility, I need a laptop having [component]",
    "As my requirements evolve, I'll need a laptop with [component]",
    "While my needs expand, I'm seeking a laptop that has [component]",
    "As technology advances, I prefer laptops featuring [component]",
    "With growing demands, I require laptops containing [component]",
    "As my skills develop, I want laptops equipped with [component]",
    "While standards rise, I need laptops that include [component]",
    "As software requirements increase, I prefer laptops offering [component]",
    "With changing work patterns, I'm looking at laptops with [component]",
    "As my projects scale up, I need laptops that provide [component]",
    "While my usage intensifies, I want a laptop having [component]",
    "The situation demands a laptop with [component]",
    "This particular project requires a laptop that has [component]",
    "The circumstances call for a laptop featuring [component]",
    "This deadline necessitates a laptop containing [component]",
    "The client expects work done on a laptop equipped with [component]",
    "This contract requires delivery using a laptop that includes [component]",
    "The specifications mandate a laptop offering [component]",
    "This role comes with a laptop with [component]",
    "The position depends on a laptop that provides [component]",
    "This task is impossible without a laptop having [component]",
    "Eventually, I'll invest in a laptop with [component]",
    "Ultimately, my goal is a laptop that has [component]",
    "In due course, I'll acquire a laptop featuring [component]",
    "At some point, I'll need a laptop containing [component]",
    "In the near future, I'll purchase a laptop equipped with [component]",
    "Before long, I intend to own a laptop that includes [component]",
    "When the time is right, I'll buy a laptop offering [component]",
    "In the coming months, I plan to get a laptop with [component]",
    "When funds permit, I'll secure a laptop that provides [component]",
    "As soon as possible, I'll obtain a laptop having [component]",
]

brand_phrases = [
    "I want laptop brand [sub brand]",
    "I'm looking for a [sub brand] laptop",
    "I'd like to purchase a laptop from [sub brand]",
    "I'm interested in buying a [sub brand] laptop",
    "I'm in the market for a [sub brand] computer",
    "I prefer laptops made by [sub brand]",
    "I'm seeking a portable computer from [sub brand]",
    "I wish to acquire a [sub brand] notebook",
    "I need a laptop manufactured by [sub brand]",
    "I'm searching for [sub brand]'s laptops",
    "My preference is for [sub brand] laptops",
    "I desire a notebook computer from [sub brand]",
    "I'm planning to get a [sub brand] laptop",
    "I want to buy a computer from the [sub brand] line",
    "I'm aiming to own a [sub brand] laptop",
    "I specifically want a [sub brand] laptop model",
    "I'm considering purchasing a [sub brand] laptop",
    "I'm after a laptop from [sub brand]",
    "I intend to acquire a [sub brand] laptop",
    "My choice for a laptop is [sub brand]",
    "I'm partial to laptops from [sub brand]",
    "I favor [sub brand] for my next laptop",
    "I'd prefer a [sub brand] for my laptop purchase",
    "I'm hoping to find a laptop by [sub brand]",
    "I want to invest in a [sub brand] laptop",
    "I'm targeting a laptop from the [sub brand] series",
    "I'm scouting for a [sub brand] laptop",
    "I'm set on getting a [sub brand] laptop",
    "I have my eye on a [sub brand] laptop",
    "I'm drawn to laptops from [sub brand]",
    "I'm keen on obtaining a [sub brand] laptop",
    "A [sub brand] laptop is what I want",
    "My goal is to purchase a [sub brand] laptop",
    "I'd like information on [sub brand] laptops",
    "I'm focused on [sub brand] for my laptop needs",
    "I'm requesting details about [sub brand] laptops",
    "Can you help me find a [sub brand] laptop?",
    "Do you carry laptops from [sub brand]?",
    "Are [sub brand] laptops available for purchase?",
    "Which [sub brand] laptop models do you recommend?",
]

anythings = [
    "whatever",
    "anything at all",
    "any single thing",
    "any item",
    "any object",
    "no matter what",
    "whatever you want",
    "whatever comes to mind",
    "anything goes",
    "whatsoever",
    "any",
    "whichever",
    "any old thing",
    "every single thing",
    "each thing",
    "all kinds of things",
    "what you please",
    "whatever is available",
    "any and all"
    "I am not sure",
    "I have no idea",
    "I have not got a clue",
    "I cannot say",
    "It escapes me",
    "I am uncertain",
    "I cannot figure it out",
    "I am clueless",
    "I have not the foggiest",
    "Who knows"
    "I do not know",
]

brand_responses = [
    "🖥️ Fantastic choice! What screen size fits your needs best?",
    "📏 Excellent! Could you specify the screen size you prefer?",
    "🎯 Great pick! What's your ideal screen size?"
]

screen_size_responses = [
    "🖼️ Nice! What screen resolution are you looking for?",
    "🔍 Perfect! Do you have a specific resolution in mind?",
    "✨ Great! Which resolution would be most comfortable for you?"
]

resolution_responses = [
    "🎨 Wonderful! What type of display do you favor (IPS, OLED, etc.)?",
    "💻 Excellent! Could you share your preferred display type?",
    "🔎 Perfect! Which display type suits you best?"
]

display_type_responses = [
    "⚡ Awesome! What refresh rate are you aiming for?",
    "🕒 Good choice! Can you tell me the refresh rate you prefer?",
    "🚀 Great! Which refresh rate do you desire for smooth visuals?"
]

refresh_rate_responses = [
    "💾 Fantastic! How much RAM would you like in your laptop?",
    "🧩 Excellent choice! What RAM capacity are you considering?",
    "🔧 Great! Could you specify the amount of RAM you need?"
]
ram_responses = [
    "⚙️ Superb! Which CPU would you prefer?",
    "🧠 Excellent! Do you have a particular CPU in mind?",
    "🚀 Great! What type of processor are you looking for?"
]

cpu_responses = [
    "🎮 Fantastic! What GPU specifications are you interested in?",
    "🖥️ Excellent! Could you indicate your preferred GPU model?",
    "🌟 Great! Which GPU would best suit your needs?"
]

gpu_responses = [
    "💼 Wonderful! What will you primarily use this laptop for?",
    "🔍 Great! Could you tell me the main purpose of your work?",
    "🚀 Excellent! What tasks or work will the laptop be used for?"
]

use_for_responses = [
    "💰 What is your budget for this purchase?",
    "💵 Could you please let me know your spending limit?",
    "🤑 How much money are you planning to invest?",
    "💸 May I know your budget range?",
    "🎯 What amount have you set aside for this?",
    "🔍 Can you share the budget you have in mind?"
]


templates = list(
    set(
        [
            temp.replace("[sub_brand]", "").replace("  ", "").strip()
            for temp in assembler()["templates"]
        ]
    )
)

use_for = assembler()["use case"]

for temp in phrases:
    templates.append(temp)

def generate_prompts(templates, components):
    template_list = [
        template.replace("[component]", component).replace("[sub brand]", component)
        for component in components
        for template in templates
    ]

    return components + template_list + anythings

def igenerate(save: bool = True, save_dir: str = None) -> dict:
    """
    Generate a intents.json file containing all the possible intents and
    patterns, and save it to a given directory.

    Args:
        save (bool): Whether to save the file to disk. Defaults to True.
        save_dir (str): The directory to save the file to. Defaults to None.

    Returns:
        dict: The generated intents.json file as a dictionary.
    """
    tag_patterns = {
        "brand": generate_prompts(brand_phrases, blst()),
        "gpu": generate_prompts(templates, glst()),
        "cpu": generate_prompts(templates, clst()),
        "ram": generate_prompts(templates, rlst()),
        "refresh rate": generate_prompts(templates, rrlst()),
        "resolution": generate_prompts(templates, srlst()),
        "display type": generate_prompts(templates, dtlst()),
        "screen size": generate_prompts(templates, sslst()),
        "use_for": use_for,
    }
    tag_responses = {
        "brand": brand_responses,
        "screen size": screen_size_responses,   
        "resolution": resolution_responses,     
        "display type": display_type_responses, 
        "refresh rate": refresh_rate_responses,    
        "ram": ram_responses,  
        "cpu": cpu_responses,                               
        "gpu": gpu_responses,
        "use_for": use_for_responses,
    }
    
    for tag, patterns in tag_patterns.items():
        new_intent = {
            "tag": tag,
            "patterns": patterns,
            "responses": tag_responses.get(tag)
        }
        intents["intents"].append(new_intent)
    
    if save:
        with open(save_dir, "w", encoding="utf-8") as f:
            json5.dump(intents, f, indent=4, ensure_ascii=False)
    
    return intents

def igenerate_lite(save: bool = True, save_dir: str = None) -> dict:
    """
    Generate a intents.json file containing all the possible intents and
    patterns, and save it to a given directory.

    Args:
        save (bool): Whether to save the file to disk. Defaults to True.
        save_dir (str): The directory to save the file to. Defaults to None.

    Returns:
        dict: The generated intents.json file as a dictionary.
    """
    tag_patterns = {
        "brand": blst(),
        "gpu": glst(),
        "cpu": clst(),
        "ram": rlst(),
        "refresh rate":rrlst(),
        "resolution": srlst(),
        "display type": dtlst(),
        "screen size": sslst(),
        "use_for": use_for,
    }
    tag_responses = {
        "brand": brand_responses,
        "screen size": screen_size_responses,   
        "resolution": resolution_responses,     
        "display type": display_type_responses, 
        "refresh rate": refresh_rate_responses,    
        "ram": ram_responses,  
        "cpu": cpu_responses,                               
        "gpu": gpu_responses,
        "use_for": use_for_responses,
    }
    
    for tag, patterns in tag_patterns.items():
        new_intent = {
            "tag": tag,
            "patterns": patterns,
            "responses": tag_responses.get(tag)
        }
        intents["intents"].append(new_intent)

    if save:
        with open(save_dir, "w", encoding="utf-8") as f:
            json5.dump(intents, f, indent=4, ensure_ascii=False)
    return intents
