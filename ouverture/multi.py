class Agent:
    def __init__(self, case):
        self.case = case

    def deplacer(self, puzzle):
        # Trouver la case finale
        case_finale = puzzle.get_case(self.case.num)
        position_finale = case_finale.position

        # Vérifier si la case est déjà à la position finale
        if self.case.position == position_finale:
            return

        # Si la case est sur la même ligne que la position finale, déplacer horizontalement
        if self.case.position[0] == position_finale[0]:
            if self.case.position[1] < position_finale[1]:
                direction = 1
            else:
                direction = -1
            for i in range(self.case.position[1], position_finale[1], direction):
                puzzle.echanger_cases((self.case.position[0], i), (self.case.position[0], i + direction))

        # Sinon, sauter jusqu'à la position finale
        else:
            if self.case.position[1] < position_finale[1]:
                x_direction = 1
            else:
                x_direction = -1
            if self.case.position[0] < position_finale[0]:
                y_direction = 1
            else:
                y_direction = -1

            # Vérifier si la case finale est sur la même colonne que la case courante
            if self.case.position[1] == position_finale[1]:
                # Sauter sur la même colonne
                for i in range(self.case.position[0], position_finale[0], y_direction):
                    puzzle.echanger_cases((i, self.case.position[1]), (i + y_direction, self.case.position[1]))
            else:
                # Sauter sur une colonne différente
                while self.case.position != position_finale:
                    # Sauter horizontalement
                    for i in range(self.case.position[1], position_finale[1], x_direction):
                        puzzle.echanger_cases((self.case.position[0], i), (self.case.position[0], i + x_direction))
                        if self.case.position == position_finale:
                            return
                    # Sauter verticalement
                    puzzle.echanger_cases((self.case.position[0], position_finale[1]), (self.case.position[0] + y_direction, position_finale[1]))

