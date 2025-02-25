from openai import OpenAI
import json
import os
from pytao import Tao

tools=[
    {
        "type": "function",
        "function": {
            "name": "execute_tao_command",
            "description": "Executes a command using Tao and returns the result. Use 'help' for available commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute via Tao."
                    }
                },
                "required": ["command"]
            }
        }
    }
]
class TaoChatSession:
    def __init__(self):
        self.messages = [
            {
                "role": "system",
                "content": "You are an assistant that interacts with Tao simulation software."
            }
        ]
        self.tao = Tao('-lat /nfs/user/nw285/wake/thin.bmad -noplot')  # Persistent Tao instance
        self.tools = tools
        self.client = OpenAI()
    
    def execute_tao_command(self, command: str) -> str:
        print(f"OpenAI called {command}\n")
        try:
            result = self.tao.cmd(command)
        except Exception as e:
            result = f"Error executing command: {str(e)}"
        return result
	
    def chat(self):
        while True:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto"
            )
            self.messages.append(completion.choices[0].message)
            tool_calls = completion.choices[0].message.tool_calls
            if tool_calls:
                for tool_call in completion.choices[0].message.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    result = self.execute_tao_command(args['command'])
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": '\n'.join(result)
                    })
            else:
                break
        print(completion.choices[0].message.content)
        
    def run(self):
        print("Interactive Tao Chat Session. Type 'exit' or 'quit' to end.")
        while True:
            user_input = input("\nUser: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Exiting interactive session.")
                break
            self.messages.append({"role": "user", "content": user_input})
            self.chat()
            
            
if __name__ == "__main__":
    session = TaoChatSession()
    session.run()


