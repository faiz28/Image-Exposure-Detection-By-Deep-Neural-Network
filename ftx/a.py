def main(words, maxWidth):
    str_len = 0
    result = ""
    List_str = []
    for word in words:
        if str_len >= maxWidth:
            print(str_len)

            for str in List_str:
                result += str

                print(str)
            str_len = len(List_str[len(List_str)-1])

            List_str.append(List_str[len(List_str)-1])
            str_len += len(word)
            List_str.append(word)
            print(str_len)
        else:
            str_len += len(word)+2

            List_str.append(word)

        print(word)


if __name__ == "__main__":
    words = ["This", "is", "an", "example", "of", "text", "justification."]
    maxWidth = 16
    main(words, maxWidth)
