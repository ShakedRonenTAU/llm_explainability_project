def prompt1():
    return (
        """The Amazon rainforest is the largest tropical rainforest in the world, 
        covering about 5.5 million square kilometers. It is home to over 10% of 
        the world’s known species, including rare animals like jaguars and harpy 
        eagles. The rainforest plays a crucial role in absorbing carbon dioxide, 
        helping to mitigate climate change. It stretches across nine countries, 
        with the majority in Brazil. Despite its importance, deforestation poses 
        a major threat to its ecosystem and biodiversity.
        Question:
        Which animal, mentioned in the text, is a rare species found in the Amazon rainforest?"""
    )


def prompt2():
    return (
        """Mount Everest, the tallest mountain on Earth, stands at 8,848 meters above 
        sea level. It is located in the Himalayas on the border of Nepal and Tibet. 
        Every year, hundreds of climbers attempt to reach its summit despite the 
        harsh conditions. The first confirmed ascent was in 1953 by Sir Edmund 
        Hillary and Tenzing Norgay. Due to climate change, the ice and glaciers 
        on Everest are melting rapidly.
        Question:
        Who were the first confirmed climbers to reach the summit of Mount Everest?"""
    )


def prompt3():
    return (
        """The Great Barrier Reef is the largest coral reef system in the world, located 
        off the coast of Queensland, Australia. It stretches over 2,300 kilometers 
        and is visible from space. The reef is home to thousands of marine species, 
        including sharks, rays, and sea turtles. Coral bleaching, caused by rising 
        ocean temperatures, threatens its survival. Conservation efforts are ongoing 
        to protect this natural wonder.
        Question:
        What phenomenon, caused by rising temperatures, threatens the Great Barrier Reef?"""
    )


def prompt4():
    return (
        """The Sahara is the largest hot desert in the world, covering approximately 
        9.2 million square kilometers. It spans across North Africa, including 
        countries like Algeria, Libya, and Egypt. Despite its extreme conditions, 
        the desert is home to diverse wildlife, including camels and desert foxes. 
        The Sahara is known for its sand dunes, which can reach heights of up to 
        180 meters. Climate change and human activities are causing the desert 
        to expand.
        Question:
        What types of animals are adapted to live in the Sahara Desert?"""
    )


def prompt5():
    return (
        """The Pacific Ocean is the largest and deepest ocean on Earth, covering more 
        than 63 million square miles. It contains over 25,000 islands, many of 
        which are volcanic. The ocean is crucial for global weather patterns and 
        climate. The Mariana Trench, located in the Pacific, is the deepest point 
        on Earth, reaching about 11 kilometers below sea level. The Pacific Ocean 
        is also a significant pathway for marine trade.
        Question:
        What is the name of the deepest point in the Pacific Ocean?"""
    )


def prompt6():
    return (
        """Antarctica, the coldest continent on Earth, has temperatures that can drop 
        below -80°C. It contains about 70% of the world's freshwater, stored in its 
        vast ice sheets. Despite the extreme cold, some unique organisms, like 
        penguins and seals, have adapted to survive there. Scientific research 
        stations are established on the continent to study climate and wildlife. 
        Ice melting in Antarctica contributes to global sea level rise.
        Question:
        Which animals have adapted to survive in the extreme cold of Antarctica?"""
    )


def prompt7():
    return (
        """Venus, the second planet from the Sun, has a surface temperature that can 
        reach up to 475°C, hotter than Mercury, even though it is further from the 
        Sun. Venus has a thick atmosphere rich in carbon dioxide, creating a strong 
        greenhouse effect. Its atmosphere contains clouds of sulfuric acid. Unlike 
        most planets, Venus rotates backward, making its day longer than its year. 
        No life as we know it can survive there due to its harsh conditions.
        Question:
        What makes Venus’s rotation unique compared to most other planets?"""
    )


def prompt8():
    return (
        """The Amazon River, the second longest river in the world, flows over 6,400 
        kilometers across South America. It has the largest drainage basin of any 
        river, covering nearly 7 million square kilometers. The river carries more 
        water than any other river on Earth. During the rainy season, it can expand 
        up to 30 miles wide in some areas. The Amazon River supports diverse 
        ecosystems and is vital for the region's communities.
        Question:
        How wide can the Amazon River expand during the rainy season?"""
    )


def prompt9():
    return (
        """Jupiter is the largest planet in our Solar System, with a diameter of about 
        142,000 kilometers. It has a Great Red Spot, a massive storm larger than 
        Earth, which has been raging for at least 300 years. Jupiter has at least 
        79 moons, including the four largest known as the Galilean moons. The planet 
        is mainly composed of hydrogen and helium. Due to its strong magnetic field, 
        it has a complex system of radiation belts.
        Question:
        What is the name of the large storm on Jupiter that has existed for over 300 years?"""
    )


def prompt10():
    return (
        """The Grand Canyon, located in Arizona, USA, is one of the deepest canyons in 
        the world, with depths up to 1,800 meters. It was formed by the Colorado 
        River cutting through layers of rock over millions of years. The canyon 
        reveals nearly 2 billion years of Earth's geological history. It is a 
        popular tourist destination, attracting millions of visitors each year. The 
        Grand Canyon is also home to unique plant and animal species.
        Question:
        How deep can the Grand Canyon reach in certain areas?"""
    )


def gather_all_prompts():
    return [
        prompt1(), prompt2(), prompt3(), prompt4(), prompt5(),
        prompt6(), prompt7(), prompt8(), prompt9(), prompt10()
    ]