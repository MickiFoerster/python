from pycurl import Curl

file_name = 'output.json'

# Opening the file in write mode
with open(file_name, 'wb') as f:
    
    # Creating a Curl instance
    c = Curl()

    # Set request options
    c.setopt(c.URL, 'https://jsonplaceholder.typicode.com/users/1')
    c.setopt(c.HTTPHEADER, ['Accept: application/json'])
    c.setopt(c.WRITEDATA, f)

    # Make the request
    c.perform()

    # Close the connection
    c.close()

    print(f'Wrote output to {file_name}')

