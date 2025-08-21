with open('cat-facts.txt', 'r', encoding='utf-8-sig') as file:
    dataset = file.readlines()
print(f'Loaded {len(dataset)} entries')
