Okay so first understand the overall idea of my project, what I am trying to build. Now I need a complete modular code for module 1 only, like all the .py files which i will add into the module 1. Now for this module firstly, I need all file sun commented no, comments in the files themseleves.

What this module should be doing will be described in the latex I have atatched.

1. For the LLMs, I would need you to use hugging face models because we neeed memory atleast for the specific session, because we need a response from an LLm then dedect sycophany and then remove the sycphancy adn then give the response.

2. Also for parsing and summarizing i want you to use langchain agents. If the given contetx by an LLM already conatsin keywords whihc are iased or may invoke sycophancy, then before an LLM generates response we provide an unbbiased context to it.

3. Similarly, if we have context history then we woudl use lang chain agents to summarize. Also if the hugging face agent does not store conversation history use the .json file and langchains to repeatedly store conversation history.


