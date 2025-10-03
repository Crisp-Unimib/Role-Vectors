import aiohttp
import asyncio
import json
import time

class Generator():

    def __init__(self, model: str, openrouter_key: str, providers: list, temperature: float = 1.0):
        """
        Initializes the generator with the specified model, OpenRouter key, temperature, and providers.
        Args:
            model (str): The name or path of the pre-trained model to use.
            openrouter_key (str): The API key for OpenRouter.
            temperature (float, optional): The temperature to use for sampling. Defaults to 0.9.
            providers (list, optional): A list of providers to use. Defaults to None.
        """
        self.model = model
        self.openrouter_key = openrouter_key
        self.temperature = temperature
        self.providers = providers if providers else []
    
    async def generate(self, prompt: str) -> str:
        """
        Asynchronously generates a response based on the given prompt using the specified model.
        """
        out = None
        async with aiohttp.ClientSession() as session:
            while out is None:
                payload = json.dumps({
                    'model': self.model,
                    'providers': {"order": self.providers},
                    'temperature': self.temperature,
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt,
                        }
                    ],
                })

                headers = {
                    'Authorization': f'Bearer {self.openrouter_key}'
                }

                try:
                    url = 'https://openrouter.ai/api/v1/chat/completions'
                    async with session.post(url, headers=headers, data=payload) as response:
                        if response.status == 200:
                            try:
                                response_json = await response.json()
                                tryout = response_json['choices'][0]['message']['content']
                                tryout = tryout.strip().strip("b'").strip()
                                if tryout.startswith('An error occurred:') or tryout == '':
                                    print('Error? Retrying...')
                                    print(tryout)
                                    await asyncio.sleep(0.5)
                                    continue
                                out = tryout
                            except:
                                out = str(await response.text())
                        else:
                            print(response.status)
                            await asyncio.sleep(0.5)
                except:
                    print('Error in sending post request? Retrying...')
                    await asyncio.sleep(0.5)
                    continue

        return out

