class Aircraft():
    aircraft_type = 'Aircraft'
    def __init__(self, fuel_level: float = 1.0, phoschek_level: float = 1.0, curr_direction: str = 'N', 
                     dropping_phoschek: bool = False, location: list = [1, 1]):
        self.fuel_level = fuel_level
        self.phoschek_level = phoschek_level
        self.curr_direction = curr_direction
        self.dropping_phoschek = dropping_phoschek
        self.location = location

    def __repr__(self):
        print_str = f'''
                      {self.aircraft_type} fuel level: {round(self.fuel_level * 100, 2)}% 
                      {self.aircraft_type} phoschek level: {round(self.phoschek_level * 100, 2)}%
                      {self.aircraft_type} current direction: {self.curr_direction}
                      {self.aircraft_type} currently dropping phoschek: {self.dropping_phoschek}
                      {self.aircraft_type} location: {self.location}
                      '''
        print(print_str)
        
class Helicopter(Aircraft):
        aircraft_type = 'Helicopter'
        def __init__(self, fuel_level: float = 1.0, phoschek_level: float = 1.0, curr_direction: str = 'N', 
                     dropping_phoschek: bool = False, location = [1, 1]):
            super().__init__(fuel_level, phoschek_level, curr_direction, dropping_phoschek, location)


class Plane(Aircraft):
        aircraft_type = 'Plane'
        def __init__(self, fuel_level: float = 1.0, phoschek_level: float = 1.0, curr_direction: str = 'N', 
                     dropping_phoschek: bool = False, location = [1, 1]):
            super().__init__(fuel_level, phoschek_level, curr_direction, dropping_phoschek, location)

# some driver code to test the classes 
x = Helicopter(fuel_level=0.85, phoschek_level=1.0, curr_direction='E', dropping_phoschek=True, location=[1,1])
x.__repr__()
