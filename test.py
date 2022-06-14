class PositionalEncoding_posi(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=32):
        super(PositionalEncoding_posi, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        div_term = torch.exp(torch.arange(d_model,0, -1).float() * (-math.log(10000.0) / d_model))
#div_term = torch.exp(torch.arange(0,d_model, 1).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
        self.register_buffer('pe', pe)

    def Posi(self,action):
        x_count = 0
        y_count = 0
        count = 0
        print(action)
        for i in action:
            if i[0] == 1:
                x_count += 1
                self.pe[count] = torch.sin(x_count * self.div_term)
            elif i[0] == -1:
                x_count -= 1
                self.pe[count] = torch.sin(x_count * self.div_term)
            elif i[1] == 1:
                y_count += 1
                self.pe[count] = torch.cos(y_count * self.div_term)
            elif i[1] == -1:
                y_count -= 1
                self.pe[count] = torch.cos(y_count * self.div_term)
            else:
                pass
            count += 1
        return self.pe

    def forward(self, x, action):
        po = self.Posi(action)
        x = x + po
        return x
