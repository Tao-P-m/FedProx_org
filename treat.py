import numpy as np

day_passed = [["Alice", 1, "Bob"],
                ["Chuck", 1, "Dave"],
                ["Dave", 1, "Eve"]]

day_passed = [["Alice", 2, "Bob", "Chuck"],
                ["Chuck", 1, "Bob"],
                ["Dave", 2, "Alice", "Bob"],
                ["Alice", 1, "Eve"]]

def main():
    # get all lines from input parser
    if len(day_passed) == 0:
        return
    name_dict = {}
    for day_ind, case in enumerate(day_passed):
        person_paid = case[0]
        cnt = case[1]
        people_received = case[2:]
        if person_paid not in name_dict.keys():
            name_dict[person_paid] = cnt
        else:
            name_dict[person_paid] += cnt
        for person_received in people_received:
            print(person_received)
            if person_received not in name_dict.keys():
                name_dict[person_received] = -1
            else:
                name_dict[person_received] -= 1
    meals = list(name_dict.values())
    if np.sum(meals) != 0:
        # not being zero-sum
        return 
    print(meals)
    min_meals = list(filter(lambda x: x > 0, meals)) # minimum meals must be bought
    max_meals = list(filter(lambda x: x < 0, meals)) # maximum meals can be offered
    print("minimum meals must be bought: {}".format(np.sum(min_meals)))

    max_payee = max(meals)
    print("minimum days to reach this point: {}".format(max_payee))
    return

if __name__ == "__main__":
    main()